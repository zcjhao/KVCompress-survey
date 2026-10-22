"""Shared helpers: seeding, press selection, environment metadata, KV-size arithmetic."""

import importlib.metadata
import platform
import random
import subprocess

import numpy as np
import torch

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_press(name: str, compression_ratio: float):
    """Return a kvpress press, or None for the uncompressed baseline.

    Add each method the team compares here so every experiment runs through the same code path,
    e.g. "snapkv": kvpress.SnapKVPress(compression_ratio=compression_ratio).
    """
    if name == "none":
        return None
    raise ValueError(f"Unknown press '{name}'")


def kv_bytes_per_token(config, dtype_bytes: int = 2) -> int:
    """Theoretical uncompressed KV-cache size per token: layers x kv_heads x head_dim x 2 (K and V) x bytes."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return config.num_hidden_layers * config.num_key_value_heads * head_dim * 2 * dtype_bytes


def cache_bytes(cache) -> int:
    """Bytes actually held by a DynamicCache (shrinks once a press evicts tokens)."""
    total = 0
    for layer in cache.layers:
        for t in (layer.keys, layer.values):
            if t is not None:
                total += t.numel() * t.element_size()
    return total


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def env_info() -> dict:
    """Everything needed to report the setup in the paper (GPU, driver, package versions, code commit)."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        driver = "unknown"
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if torch.cuda.is_available()
        else 0,
        "driver": driver,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _pkg_version("transformers"),
        "kvpress": _pkg_version("kvpress"),
        "git_commit": commit,
    }
