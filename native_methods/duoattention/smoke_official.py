"""Stage-2 smoke test using MIT's real inference patch (not kvpress masking).

Run ONLY after the official, separate Python environment is installed and validated.
This is NOT a RULER comparison nor proof of physical memory savings.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
from duo_attn.patch import enable_duo_attention_eval


ROOT = Path(__file__).resolve().parents[2]
PATTERN = ROOT / "assets" / "duoattention_pattern"
EXPECTED_PATTERN_SHA = "042e0c23e89278525928bf45b6eba3200d9bf4a65231cd469d69eb9b98c61218"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--sparsity", type=float, default=0.5, help="Fraction of KV-head groups designated streaming; not an exact byte budget")
    p.add_argument("--prompt-tokens", type=int, default=512)
    args = p.parse_args()
    assert 0 <= args.sparsity < 1
    assert 0 < args.prompt_tokens <= 4096

    actual_sha = hashlib.sha256((PATTERN / "full_attention_heads.tsv").read_bytes()).hexdigest()
    if actual_sha != EXPECTED_PATTERN_SHA:
        raise RuntimeError(f"Head pattern hash mismatch: {actual_sha}")

    np.random.seed(0)
    torch.manual_seed(0)
    scores, sink, recent = load_attn_pattern(str(PATTERN))
    selected, actual_sparsity = sparsify_attention_heads(scores.copy(), sparsity=args.sparsity)
    print(f"KV-head pattern shape={selected.shape}, streaming ratio={actual_sparsity:.4f}, sink={sink}, recent={recent}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager", local_files_only=True,
    ).eval()
    enable_duo_attention_eval(model, selected, sink_size=sink, recent_size=recent)
    model = model.cuda()

    text = "The factual code is BLUE. " * (args.prompt_tokens // 6 + 1)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids[:, :args.prompt_tokens].cuda()
    if ids.shape[1] < 4:
        raise RuntimeError("Too few prompt tokens")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    print("Generated tokens:", output.shape[1] - ids.shape[1])
    print("CUDA peak allocated GB:", round(torch.cuda.max_memory_allocated() / 1e9, 3))
    print("Native smoke forward passed; physical KV-byte reduction still requires explicit verification.")


if __name__ == "__main__":
    main()
