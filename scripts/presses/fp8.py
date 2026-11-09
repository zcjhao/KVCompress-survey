"""FP8 (E4M3) KV cache: every token kept at 1 byte per entry, half of BF16.

Not a press: build() returns no press and build_cache() returns the FP8 cache (scripts/kv_quant.py).
Scale fixed at 1.0, as vLLM's kv_cache_dtype="fp8" without calibration, so this harness's accuracy
should match what vLLM serves. The A100 has no FP8 math; the cache is stored in FP8 and converted to
BF16 for attention. Run with RATIO=0.5.
"""
from kv_quant import FP8Layer, QuantizedKVCache


def _check(compression_ratio: float):
    if compression_ratio != 0.5:
        raise ValueError(f"fp8 halves the cache: use RATIO=0.5, got {compression_ratio}")


def build(compression_ratio: float):
    _check(compression_ratio)
    return None


def build_cache(compression_ratio: float, config):
    _check(compression_ratio)
    return QuantizedKVCache(config, FP8Layer)
