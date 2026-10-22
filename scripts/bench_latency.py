"""Prefill latency (TTFT), per-token decode latency (TPOT), peak memory and KV-cache size vs. context length.

Uses random token ids: speed and memory don't depend on the text, only on its length.

    python scripts/bench_latency.py --lengths 1024 2048 4096 8192 16384
"""

import argparse
import contextlib
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from common import DEFAULT_MODEL, build_press, cache_bytes, env_info, kv_bytes_per_token, set_seed


@torch.inference_mode()
def run_once(model, press, length: int, decode_tokens: int, vocab_size: int) -> dict:
    input_ids = torch.randint(0, vocab_size, (1, length), device=model.device)
    cache = DynamicCache()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with press(model) if press is not None else contextlib.nullcontext():
        out = model(input_ids=input_ids, past_key_values=cache, logits_to_keep=1)
    next_tok = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    kv_after_prefill = cache_bytes(cache)

    # Pass positions explicitly: after eviction the cache holds fewer entries than the true sequence length.
    t0 = time.perf_counter()
    for i in range(decode_tokens):
        pos = torch.tensor([[length + i]], device=model.device)
        out = model(input_ids=next_tok, past_key_values=cache, position_ids=pos, logits_to_keep=1)
        next_tok = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    tpot = (time.perf_counter() - t0) / decode_tokens

    return {
        "ttft_s": ttft,
        "tpot_ms": tpot * 1e3,
        "kv_cache_gb": kv_after_prefill / 1e9,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--press", default="none")
    p.add_argument("--compression-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = p.parse_args()

    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0"
    ).eval()
    press = build_press(args.press, args.compression_ratio)
    vocab = model.config.vocab_size
    weights_gb = torch.cuda.memory_allocated() / 1e9

    run_once(model, press, 512, 8, vocab)  # warm-up: CUDA context, kernel selection

    rows = []
    for length in args.lengths:
        try:
            trials = [run_once(model, press, length, args.decode_tokens, vocab) for _ in range(args.repeats)]
        except torch.cuda.OutOfMemoryError:
            print(f"len={length}: out of memory, stopping")
            rows.append({"length": length, "oom": True})
            break
        row = {"length": length, **{k: statistics.median(t[k] for t in trials) for k in trials[0]}}
        rows.append(row)
        print(
            f"len={length:6d}  TTFT {row['ttft_s']:.3f} s  TPOT {row['tpot_ms']:.1f} ms  "
            f"KV {row['kv_cache_gb']:.2f} GB  peak {row['peak_memory_gb']:.2f} GB"
        )

    model_tag = args.model.split("/")[-1]
    out_dir = Path(args.out) / f"latency_{model_tag}_{args.press}_r{args.compression_ratio}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args),
        "batch_size": 1,
        "weights_gb": round(weights_gb, 2),
        "kv_bytes_per_token_uncompressed": kv_bytes_per_token(model.config),
        "results": rows,
        "env": env_info(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
