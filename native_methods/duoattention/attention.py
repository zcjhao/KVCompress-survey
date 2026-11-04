"""Functional port of DuoAttention head-level *physical* dual KV caches.

Works with Transformers 5.2.0 LlamaAttention.forward under the team's existing
Python/PyTorch/CUDA environment. NOT the author's optimized kernel and not
validated on the user's A100 yet. Inference only; one unpadded request at a time.
"""
from __future__ import annotations
from contextlib import contextmanager
from types import MethodType
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from .dual_cache import DualKVCache, append_layer, _cat


def _q_indices(kv_idx: torch.Tensor, group_size: int) -> torch.Tensor:
    offsets = torch.arange(group_size, device=kv_idx.device)
    return (kv_idx[:, None] * group_size + offsets[None, :]).reshape(-1)


def _group_attention(q, k, v, q_pos, k_pos, *, sink: int | None, recent: int | None):
    """True causal attention. Streaming group also obeys sink/recent restriction."""
    if q.shape[1] == 0:
        return q
    kv_groups = q.shape[1] // k.shape[1]
    k = torch.repeat_interleave(k, kv_groups, dim=1)
    v = torch.repeat_interleave(v, kv_groups, dim=1)
    # Retrieval K/V are chronologically contiguous: the Q block is the final
    # q_len positions of K. Lower-right causal bias exactly represents the
    # original absolute-position mask while enabling fused CUDA FlashAttention.
    if sink is None:
        q_len, kv_len = q.shape[-2], k.shape[-2]
        if kv_len < q_len:
            raise RuntimeError('Retrieval KV length is shorter than query length')
        expected_q_positions = k_pos[-q_len:]
        if not torch.equal(q_pos, expected_q_positions):
            raise RuntimeError('Retrieval queries must be the last KV positions')
        if q.is_cuda:
            # Fail fast instead of silently falling back to a memory-heavy
            # quadratic Math SDPA kernel at 64K/128K.
            from torch.backends.cuda import SDPAParams, can_use_flash_attention
            supported = can_use_flash_attention(
                SDPAParams(q, k, v, None, 0.0, False, False), debug=True)
            if not supported:
                raise RuntimeError('FlashAttention unavailable for retrieval group; '
                                   'not running memory-heavy Math Attention')
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=causal_lower_right(q_len, kv_len),
                    dropout_p=0.0, is_causal=False)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=causal_lower_right(q_len, kv_len),
            dropout_p=0.0, is_causal=False)

    # Streaming K/V is bounded (sink + recent + incoming chunk), so an explicit
    # absolute-position mask remains tractable even with Math SDPA.
    valid = k_pos[None, :] <= q_pos[:, None]
    valid &= ((k_pos[None, :] < sink) |
              (k_pos[None, :] >= (q_pos[:, None] - recent + 1)))
    mask = valid[None, None, :, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                          dropout_p=0.0, is_causal=False)


class NativeDuoAttention:
    def __init__(self, model, retrieval_mask, *, sink_size: int, recent_size: int):
        import transformers
        if transformers.__version__ != '5.2.0':
            raise RuntimeError(f'This adapter is pinned to transformers==5.2.0, got {transformers.__version__}')
        if model.__class__.__name__ != 'LlamaForCausalLM':
            raise TypeError('Only LlamaForCausalLM is supported')
        if model.training:
            raise ValueError('Call model.eval() first')
        if sink_size < 1 or recent_size < 1:
            raise ValueError('sink_size and recent_size must be positive')
        n_layer, n_kv = len(model.model.layers), model.config.num_key_value_heads
        mask = torch.as_tensor(retrieval_mask, dtype=torch.bool)
        if tuple(mask.shape) != (n_layer, n_kv):
            raise ValueError(f'head pattern shape must be {(n_layer, n_kv)}, got {tuple(mask.shape)}')
        self.model = model
        self.mask = mask
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.cache = DualKVCache(n_layer)
        self._old_forward = []

    def __enter__(self):
        if self._old_forward:
            raise RuntimeError('Already patched')
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            self._old_forward.append((attn, attn.forward))
            def wrapped(module, hidden_states, position_embeddings=None, attention_mask=None,
                        past_key_values=None, cache_position=None, *, _idx=layer_idx, **kwargs):
                return self._forward(_idx, module, hidden_states, position_embeddings,
                                     cache_position)
            attn.forward = MethodType(wrapped, attn)
        return self

    def __exit__(self, exc_type, exc, tb):
        for attn, original in self._old_forward:
            attn.forward = original
        self._old_forward.clear()

    def reset(self):
        self.cache.reset()

    def _forward(self, idx, module, hidden_states, position_embeddings, cache_position):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        if hidden_states.shape[0] != 1:
            raise ValueError('Native adapter currently supports batch_size=1 only')
        if cache_position is None or position_embeddings is None:
            raise RuntimeError('Explicit cache_position and position_embeddings required')
        state = self.cache.layers[idx]
        B, T, _ = hidden_states.shape
        q_pos = cache_position.reshape(-1)
        if q_pos.numel() != T or int(q_pos[0]) != state.seen or \
           not torch.equal(q_pos, torch.arange(state.seen, state.seen + T, device=q_pos.device)):
            raise RuntimeError(f'Non-monotonic absolute positions at layer {idx}: '
                               f'cache_seen={state.seen}, positions={q_pos}')
        D = module.head_dim
        H = module.config.num_attention_heads
        KV = module.config.num_key_value_heads
        G = H // KV
        shape = (B, T, -1, D)
        q = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        retrieval_idx = torch.nonzero(self.mask[idx], as_tuple=True)[0].to(k.device)
        streaming_idx = torch.nonzero(~self.mask[idx], as_tuple=True)[0].to(k.device)
        outs = torch.empty((B, H, T, D), dtype=q.dtype, device=q.device)
        if retrieval_idx.numel():
            q_idx = _q_indices(retrieval_idx, G)
            q_r = q.index_select(1, q_idx)
            k_r, v_r = k.index_select(1, retrieval_idx), v.index_select(1, retrieval_idx)
            all_k = _cat(state.retrieval_k, k_r)
            all_v = _cat(state.retrieval_v, v_r)
            positions = torch.arange(state.seen + T, device=k.device)
            out = _group_attention(q_r, all_k, all_v, q_pos, positions, sink=None, recent=None)
            outs.index_copy_(1, q_idx, out)
        else:
            k_r, v_r = k[:, :0, :, :], v[:, :0, :, :]
        if streaming_idx.numel():
            q_idx = _q_indices(streaming_idx, G)
            q_s = q.index_select(1, q_idx)
            k_s, v_s = k.index_select(1, streaming_idx), v.index_select(1, streaming_idx)
            k_parts, v_parts, positions = [], [], []
            if state.sink_k is not None:
                k_parts.append(state.sink_k)
                v_parts.append(state.sink_v)
                positions.append(torch.arange(state.sink_k.shape[-2], device=k.device))
            if state.recent_k is not None:
                k_parts.append(state.recent_k)
                v_parts.append(state.recent_v)
                positions.append(torch.arange(state.seen - state.recent_k.shape[-2],
                                              state.seen, device=k.device))
            k_parts.append(k_s)
            v_parts.append(v_s)
            positions.append(q_pos)
            all_k = torch.cat(k_parts, dim=-2)
            all_v = torch.cat(v_parts, dim=-2)
            key_pos = torch.cat(positions)
            out = _group_attention(q_s, all_k, all_v, q_pos, key_pos,
                                   sink=self.sink_size, recent=self.recent_size)
            outs.index_copy_(1, q_idx, out)
        else:
            k_s, v_s = k[:, :0, :, :], v[:, :0, :, :]
        append_layer(state, k_r, v_r, k_s, v_s, self.sink_size, self.recent_size)
        output = outs.transpose(1, 2).reshape(B, T, H * D).contiguous()
        return module.o_proj(output), None
