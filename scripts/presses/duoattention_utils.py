"""DuoAttention-only immutable head-pattern I/O and logical budget accounting.

Kept separate from the team's ``scripts/common.py`` so other methods are unchanged.
The retained bytes here are *logical estimates*, not real allocated memory.
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "duoattention.json"


@lru_cache(maxsize=1)
def method_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def pattern_dir() -> Path:
    return ROOT / method_config()["pattern_dir"]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _frozen_pattern() -> tuple[np.ndarray, int, int]:
    path = pattern_dir()
    actual = sha256(path / "full_attention_heads.tsv")
    expected = method_config().get("pattern_sha256")
    if expected and actual != expected:
        raise ValueError(f"Frozen DuoAttention head pattern hash mismatch: {actual} != {expected}")
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    scores = np.loadtxt(path / "full_attention_heads.tsv", dtype=float, delimiter="\t")
    if scores.shape != (32, 8) or not np.isfinite(scores).all():
        raise ValueError("Frozen head pattern must have 32 layers x 8 KV heads, all finite")
    sink, recent = int(cfg["sink_size"]), int(cfg["recent_size"])
    if sink <= 0 or recent <= 0:
        raise ValueError("Frozen head pattern window sizes must be positive")
    return np.clip(scores, 0, 1), sink, recent


def read_pattern() -> tuple[np.ndarray, int, int]:
    """Copy scores to prevent accidentally mutating the cached frozen pattern."""
    scores, sink, recent = _frozen_pattern()
    return scores.copy(), sink, recent


def budget_plan(
    length: int,
    target: float,
    sink: int = 128,
    recent: int = 256,
    layers: int = 32,
    kv_heads: int = 8,
    dim: int = 128,
    bytes_per_element: int = 2,
) -> dict:
    """Allocate integer streaming heads to stay <= requested *logical* retained ratio.

    The model still holds full tensors in kvpress.DuoAttentionPress; these estimates must
    not be reported as physical memory savings. No allocator or temporary costs included.
    """
    if length <= 0 or not (0 < target <= 1):
        raise ValueError("length > 0 and retained_budget in (0, 1] are required")
    if min(sink, recent, layers, kv_heads, dim, bytes_per_element) <= 0:
        raise ValueError("Window, layers, heads, dimension and element size must be positive")
    total = layers * kv_heads
    window = min(length, sink + recent)
    minimum = window / length
    if target < minimum - 1e-12:
        raise ValueError(f"Budget {target} is below streaming-window minimum {minimum:.6f}")
    if math.isclose(minimum, 1.0):
        streaming = 0
    else:
        streaming = 0 if target == 1 else math.ceil((1 - target) / (1 - minimum) * total - 1e-10)
        streaming = min(total, max(0, streaming))
    retrieval = total - streaming
    effective = retrieval / total + streaming / total * minimum
    full_bytes = 2 * layers * kv_heads * dim * bytes_per_element * length
    return dict(
        context_tokens=length,
        target_retained_ratio=target,
        retrieval_heads=retrieval,
        streaming_heads=streaming,
        total_kv_heads=total,
        head_compression_ratio=streaming / total,
        effective_retained_ratio=effective,
        logical_context_kv_bytes=round(full_bytes * effective),
        full_context_kv_bytes=full_bytes,
        logical_kv_bytes_per_context_token=full_bytes * effective / length,
        sink_size=sink,
        recent_size=recent,
        budget_accounting="logical_context_kv",
        physical_memory_saving_claim=False,
    )


def binary_heads(scores: np.ndarray, streaming_heads: int) -> np.ndarray:
    """Select exactly N lowest-scoring heads (same tie behavior as pinned kvpress)."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (32, 8) and scores.ndim != 2:
        raise ValueError("Expected 2D scores")
    if not 0 <= streaming_heads <= scores.size:
        raise ValueError("Invalid streaming head count")
    mask = np.ones(scores.shape, dtype=float)
    indices = np.argsort(scores, axis=None)[:streaming_heads]
    mask[np.unravel_index(indices, scores.shape)] = 0
    return mask
