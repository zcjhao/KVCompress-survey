"""SnapKV (Li et al., 2024): keep the context tokens that the last window of tokens attends to most.

kvpress compresses the context before the question is appended, so the observation window is the
end of the context, not the question (see the README note on query-agnostic compression).
kvpress defaults: window_size=64, kernel_size=5.
"""
from kvpress import SnapKVPress


def build(compression_ratio: float):
    return SnapKVPress(compression_ratio=compression_ratio)
