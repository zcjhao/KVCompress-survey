"""Physical, head-grouped KV storage for a Hugging Face Llama native-method runner.

Inference only; batch=1. Retrieval KV is full history, streaming KV is
(sink prefix, nonoverlapping recent suffix). Never store the dropped middle.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


def tensor_bytes(t):
    return 0 if t is None else t.numel() * t.element_size()


@dataclass
class LayerState:
    retrieval_k: torch.Tensor | None = None
    retrieval_v: torch.Tensor | None = None
    sink_k: torch.Tensor | None = None
    sink_v: torch.Tensor | None = None
    recent_k: torch.Tensor | None = None
    recent_v: torch.Tensor | None = None
    seen: int = 0

    def bytes(self):
        return sum(tensor_bytes(getattr(self, name)) for name in (
            'retrieval_k','retrieval_v','sink_k','sink_v','recent_k','recent_v'))


class DualKVCache:
    def __init__(self, n_layers: int):
        self.layers = [LayerState() for _ in range(n_layers)]

    def reset(self):
        for i in range(len(self.layers)):
            self.layers[i] = LayerState()

    def bytes(self):
        return sum(layer.bytes() for layer in self.layers)

    def breakdown(self):
        retr = sum(tensor_bytes(t) for s in self.layers for t in (s.retrieval_k,s.retrieval_v))
        stream = sum(tensor_bytes(t) for s in self.layers for t in (s.sink_k,s.sink_v,s.recent_k,s.recent_v))
        return {'retrieval_kv_bytes': retr, 'streaming_kv_bytes': stream,
                'kv_cache_bytes': retr + stream, 'seen_tokens': self.layers[0].seen}


def _cat(old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
    return new.contiguous() if old is None else torch.cat((old, new), dim=-2).contiguous()


def append_layer(state: LayerState, k_retr: torch.Tensor, v_retr: torch.Tensor,
                 k_stream: torch.Tensor, v_stream: torch.Tensor,
                 sink_size: int, recent_size: int) -> None:
    """Append a chunk, leaving only the physically retained KV tensors.

    All tensors are [batch, kv_heads, tokens, head_dim]. Call *after* computing
    the chunk attention, whose context includes old states plus the new chunk.
    """
    n = k_retr.shape[-2] if k_retr is not None else k_stream.shape[-2]
    start = state.seen
    if k_retr.shape[1]:
        state.retrieval_k = _cat(state.retrieval_k, k_retr)
        state.retrieval_v = _cat(state.retrieval_v, v_retr)
    if k_stream.shape[1]:
        # Initial sink positions are kept once; recent tokens never duplicate them.
        n_sink = max(0, min(n, sink_size - start))
        if n_sink:
            state.sink_k = _cat(state.sink_k, k_stream[..., :n_sink, :])
            state.sink_v = _cat(state.sink_v, v_stream[..., :n_sink, :])
        non_sink_start = n_sink
        if non_sink_start < n:
            next_k = k_stream[..., non_sink_start:, :]
            next_v = v_stream[..., non_sink_start:, :]
            state.recent_k = _cat(state.recent_k, next_k)
            state.recent_v = _cat(state.recent_v, next_v)
            if state.recent_k.shape[-2] > recent_size:
                state.recent_k = state.recent_k[..., -recent_size:, :].contiguous()
                state.recent_v = state.recent_v[..., -recent_size:, :].contiguous()
    state.seen += n
