"""Quantized KV caches for Hugging Face transformers: KIVI-style int4/int2 and FP8 (E4M3).

Both keep every token and store fewer bits per entry. They are caches, not kvpress presses, so a
press file returns them from build_cache() (see presses/kivi.py, presses/fp8.py).

Shared behaviour, so the two are comparable:
- The context prefill attends to its own exact BF16 keys/values (as in KIVI and vLLM); the stored
  copy is quantized. The question and every decoded token then read the quantized cache.
- Data is stored packed (uint8 / float8), so memory is really saved; stored_bytes() reports the
  bytes actually held, including scales, zero points and the BF16 residual window.
- Each step dequantizes the whole cache to BF16 for attention (like transformers' QuantizedCache),
  so these caches save memory, not time, in this harness. Speed claims need a serving engine.

KIVI (Liu et al., ICML 2024): keys are quantized per channel, in groups of `group_size` tokens;
values per token, in groups of `group_size` channels; asymmetric min-max with round-to-nearest.
The most recent tokens (fewer than `residual_length`) stay in BF16 and are quantized
`residual_length` at a time. Defaults are the paper's: group 32, residual 128, FP16 scales.
"""

import torch
from transformers.cache_utils import Cache, DynamicLayer

FP8_MAX = 448.0  # largest finite float8_e4m3fn value


def _nbytes(t) -> int:
    return 0 if t is None else t.numel() * t.element_size()


def pack(q: torch.Tensor, nbits: int) -> torch.Tensor:
    """Pack integers in [0, 2**nbits) along the last dim, 8 // nbits per byte."""
    per_byte = 8 // nbits
    q = q.to(torch.uint8).unflatten(-1, (-1, per_byte))
    shifts = torch.arange(0, 8, nbits, dtype=torch.uint8, device=q.device)
    return (q << shifts).sum(-1, dtype=torch.uint8)


def unpack(p: torch.Tensor, nbits: int) -> torch.Tensor:
    shifts = torch.arange(0, 8, nbits, dtype=torch.uint8, device=p.device)
    return ((p.unsqueeze(-1) >> shifts) & ((1 << nbits) - 1)).flatten(-2)


def quantize(x: torch.Tensor, nbits: int, group_size: int, per_channel: bool):
    """Asymmetric min-max quantization of x [batch, heads, tokens, head_dim].

    per_channel=True (keys): one scale per channel per group of group_size tokens.
    per_channel=False (values): one scale per token per group of group_size channels.
    Returns (packed uint8 [b, h, t, d * nbits / 8], scale, zero point), scales in float16.
    """
    b, h, t, d = x.shape
    x = x.float()
    g = x.view(b, h, t // group_size, group_size, d) if per_channel else x.view(b, h, t, d // group_size, group_size)
    dim = 3 if per_channel else 4
    mn = g.amin(dim, keepdim=True)
    scale = (g.amax(dim, keepdim=True) - mn).clamp(min=1e-6) / ((1 << nbits) - 1)
    q = ((g - mn) / scale).round_().clamp_(0, (1 << nbits) - 1).view(b, h, t, d)
    return pack(q, nbits), scale.half(), mn.half()


def dequantize(packed, scale, mn, nbits: int, group_size: int, per_channel: bool, dtype) -> torch.Tensor:
    # FP16 math (exact for the integer codes) keeps the temporary half the size of FP32 at 128K.
    q = unpack(packed, nbits).half()
    b, h, t, d = q.shape
    g = q.view(b, h, t // group_size, group_size, d) if per_channel else q.view(b, h, t, d // group_size, group_size)
    return g.mul_(scale).add_(mn).view(b, h, t, d).to(dtype)


class KIVILayer(DynamicLayer):
    """One layer of a KIVI cache. self.keys / self.values hold only the BF16 residual window."""

    def __init__(self, nbits: int, group_size: int = 32, residual_length: int = 128):
        super().__init__()
        if nbits not in (2, 4):
            raise ValueError(f"nbits must be 2 or 4, got {nbits}")
        if residual_length % group_size:
            raise ValueError("residual_length must be a multiple of group_size")
        self.nbits, self.group_size, self.residual_length = nbits, group_size, residual_length
        self.n_quantized = 0
        self._k = self._v = None  # (packed, scale, zero point) of the quantized tokens

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        b, h, _, d = key_states.shape
        if d % self.group_size or d % (8 // self.nbits):
            raise ValueError(f"head_dim {d} must be a multiple of group_size {self.group_size}")
        self.keys = key_states.new_empty((b, h, 0, d))
        self.values = value_states.new_empty((b, h, 0, d))
        self.is_initialized = True

    def _dequantized(self):
        k = dequantize(*self._k, self.nbits, self.group_size, True, self.dtype)
        v = dequantize(*self._v, self.nbits, self.group_size, False, self.dtype)
        return k, v

    def _flush(self):
        """Move the oldest residual tokens into the quantized store, residual_length at a time."""
        n = self.keys.shape[-2] // self.residual_length * self.residual_length
        if n == 0:
            return
        k_new = quantize(self.keys[..., :n, :], self.nbits, self.group_size, per_channel=True)
        v_new = quantize(self.values[..., :n, :], self.nbits, self.group_size, per_channel=False)
        if self._k is None:
            self._k, self._v = k_new, v_new
        else:
            # Key scales are grouped along tokens (dim 2) just like the data; value scales are per token.
            self._k = tuple(torch.cat(p, dim=2) for p in zip(self._k, k_new))
            self._v = tuple(torch.cat(p, dim=2) for p in zip(self._v, v_new))
        self.n_quantized += n
        self.keys = self.keys[..., n:, :].clone()
        self.values = self.values[..., n:, :].clone()

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            # Context prefill: attend to the exact keys/values, store them quantized.
            self.lazy_initialization(key_states, value_states)
            self.keys, self.values = key_states, value_states
            self._flush()
            return key_states, value_states
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        if self._k is None:
            keys, values = self.keys, self.values
        else:
            k, v = self._dequantized()
            keys, values = torch.cat([k, self.keys], dim=-2), torch.cat([v, self.values], dim=-2)
        self._flush()
        return keys, values

    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        return self.n_quantized + self.keys.shape[-2]

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("KIVILayer does not support crop")

    def stored_bytes(self) -> int:
        quantized = sum(_nbytes(t) for parts in (self._k or (), self._v or ()) for t in parts)
        return quantized + _nbytes(self.keys) + _nbytes(self.values)

    def __repr__(self):
        return (f"KIVILayer(nbits={self.nbits}, group_size={self.group_size}, "
                f"residual_length={self.residual_length})")


class FP8Layer(DynamicLayer):
    """One layer of an FP8 (E4M3) cache with a fixed scale of 1.0, matching vLLM's default
    kv_cache_dtype="fp8" without calibrated scales. Values outside +-448 are clamped."""

    def update(self, key_states, value_states, cache_kwargs=None):
        k8 = key_states.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        v8 = value_states.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.keys, self.values = k8, v8
            return key_states, value_states
        self.keys = torch.cat([self.keys, k8], dim=-2)
        self.values = torch.cat([self.values, v8], dim=-2)
        return self.keys.to(self.dtype), self.values.to(self.dtype)

    def stored_bytes(self) -> int:
        return _nbytes(self.keys) + _nbytes(self.values)

    def __repr__(self):
        return "FP8Layer(float8_e4m3fn, scale=1.0)"


class QuantizedKVCache(Cache):
    """A cache with one quantized layer per decoder layer."""

    def __init__(self, config, make_layer):
        config = config.get_text_config(decoder=True)
        super().__init__(layers=[make_layer() for _ in range(config.num_hidden_layers)])

    def __repr__(self):
        return f"QuantizedKVCache({self.layers[0]!r} x {len(self.layers)})"
