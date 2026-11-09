"""KIVI-style KV quantization (Liu et al., 2024): keys per channel, values per token, recent tokens in BF16.

Not a press: every token is kept, at fewer bits, so build() returns no press and build_cache()
returns the quantized cache (scripts/kv_quant.py). The compression ratio picks the bit width by its
nominal saving over BF16, so run folders line up with eviction ratios at the same budget:
    RATIO=0.75   -> int4 (measured about 31% of BF16 with scales, i.e. 0.69 saved)
    RATIO=0.875  -> int2 (measured about 19% of BF16)
Group size 32 and a 128-token BF16 residual, as in the paper.
"""
from kv_quant import KIVILayer, QuantizedKVCache

BITS = {0.75: 4, 0.875: 2}


def _nbits(compression_ratio: float) -> int:
    if compression_ratio not in BITS:
        raise ValueError(f"kivi supports RATIO in {sorted(BITS)} (int4, int2), got {compression_ratio}")
    return BITS[compression_ratio]


def build(compression_ratio: float):
    _nbits(compression_ratio)
    return None


def build_cache(compression_ratio: float, config):
    nbits = _nbits(compression_ratio)
    return QuantizedKVCache(config, lambda: KIVILayer(nbits, group_size=32, residual_length=128))
