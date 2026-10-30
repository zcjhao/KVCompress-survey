"""Accuracy and per-sample cost on RULER through the kvpress pipeline.

The baseline is just press=none, so every compression method later runs through exactly this
code path: same model, prompts, decoding and scoring.

    python scripts/eval_ruler.py --context-length 32768 --per-task 50
    python scripts/eval_ruler.py --context-length 32768 --per-task 2    # quick smoke test

Examples come from splits/ruler<length>_s<seed>.json (made by make_splits.py and committed): a fixed
shuffled order of sample ids per task, of which the first --per-task are used. So --per-task 20 is a
subset of --per-task 50, and everyone on the team scores identical examples.

Each sample is appended to results/<run>/predictions.jsonl as soon as it finishes, so an interrupted
run (e.g. a Spot preemption) picks up where it stopped when re-launched with the same arguments, and
raising --per-task later only runs the new examples.
"""

import argparse
import ast
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import DynamicCache, pipeline

import kvpress  # noqa: F401  (registers the "kv-press-text-generation" pipeline)
from common import (
    DEFAULT_MODEL,
    REPO_ROOT,
    RULER_SOURCES,
    ForwardTimer,
    GpuEnergy,
    build_press,
    cache_bytes,
    env_info,
    kv_bytes_per_token,
    set_seed,
)

CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


def score(pred: str, answers: list[str], task: str) -> float:
    """RULER's official metrics (as in kvpress): QA tasks need any one reference
    (string_match_part); the others need every reference to appear (string_match_all)."""
    pred = CONTROL_CHARS.sub("", pred.strip()).strip().lower()
    hits = [a.lower() in pred for a in answers]
    if task.startswith("qa"):
        return float(any(hits))
    return sum(hits) / len(hits)


def load_split(context_length: int, seed: int, per_task: int) -> tuple[dict, list[int]]:
    path = REPO_ROOT / "splits" / f"ruler{context_length}_s{seed}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. The split files are fixed and committed (seed 0 at 32K, 64K "
                         "and 128K); use --seed 0 and don't regenerate them, or results stop being comparable.")
    split = json.loads(path.read_text())
    ids = [i for order in split["order"].values() for i in order[:per_task]]
    return split, sorted(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--context-length", type=int, default=32768, choices=sorted(RULER_SOURCES))
    p.add_argument("--per-task", type=int, default=50, help="examples per RULER task, taken from the split file")
    p.add_argument("--press", default="none")
    p.add_argument("--compression-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    set_seed(args.seed)
    split, ids = load_split(args.context_length, args.seed, args.per_task)
    repo, revision = RULER_SOURCES[args.context_length]
    if (split["dataset"], split["revision"]) != (repo, revision):
        raise SystemExit(f"Split file was made from {split['dataset']}@{split['revision']}, not {repo}@{revision}")

    # The run name has no --per-task, so raising it later extends the same run instead of starting over.
    model_tag = args.model.split("/")[-1]
    run = f"ruler{args.context_length}_{model_tag}_{args.press}_r{args.compression_ratio}_s{args.seed}"
    out_dir = Path(args.out) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        # A preemption can leave a torn last line (often zero bytes); drop it so that example reruns.
        lines = pred_path.read_bytes().splitlines(keepends=True)
        good = []
        for line in lines:
            try:
                json.loads(line)["sample_id"]
                good.append(line)
            except (ValueError, KeyError, TypeError):
                pass
        if len(good) != len(lines):
            print(f"Dropping {len(lines) - len(good)} unreadable line(s) from {pred_path}")
            pred_path.write_bytes(b"".join(l if l.endswith(b"\n") else l + b"\n" for l in good))
        done = {json.loads(line)["sample_id"] for line in good}

    ds = load_dataset(repo, str(args.context_length), split="test", revision=revision)
    todo = [i for i in ids if i not in done]
    print(f"{run}: {len(ids)} samples ({args.per_task}/task), {len(ids) - len(todo)} already done, {len(todo)} to run")

    pipe = pipeline(
        "kv-press-text-generation",
        model=args.model,
        device="cuda:0",
        dtype=torch.bfloat16,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    press = build_press(args.press, args.compression_ratio)
    timer = ForwardTimer(pipe.model.model)
    energy = GpuEnergy()
    started = datetime.now(timezone.utc)
    t_start = time.perf_counter()

    with pred_path.open("a") as f:
        for n, i in enumerate(todo, 1):
            row = ds[i]
            answers = ast.literal_eval(row["answer"]) if isinstance(row["answer"], str) else list(row["answer"])
            cache = DynamicCache()
            timer.reset()
            torch.cuda.reset_peak_memory_stats()
            e0 = energy.read()
            t0 = time.perf_counter()
            pred = pipe(
                row["context"],
                question=row["question"],
                answer_prefix=row["answer_prefix"],
                press=press,
                cache=cache,
                max_new_tokens=int(row["max_new_tokens"]),
            )["answer"]
            latency = time.perf_counter() - t0
            e1 = energy.read()

            # calls[0]: context prefill (incl. compression); calls[1]: question + first token; rest: one per token.
            (context_tokens, prefill_s), (question_tokens, question_s) = timer.calls[:2]
            decode_s = sum(s for _, s in timer.calls[2:])
            rec = {
                "sample_id": i,
                "task": row["task"],
                "score": score(pred, answers, row["task"]),
                "context_tokens": context_tokens,
                "question_tokens": question_tokens,
                "output_tokens": len(timer.calls) - 1,
                # The pipeline trims the answer from the cache afterwards, so this is the (compressed) context only.
                "kv_cache_tokens": cache.get_seq_length(),
                "kv_cache_bytes": cache_bytes(cache),
                "prefill_s": prefill_s,
                "ttft_s": prefill_s + question_s,
                "tpot_ms": decode_s / max(len(timer.calls) - 2, 1) * 1e3,
                "latency_s": latency,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "energy_j": None if e0 is None else e1 - e0,
                "pred": pred,
                "answer": answers,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())  # so a preemption can't leave a half-written line
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  last: {latency:.1f} s, {context_tokens} context tokens")

    # One line per invocation, so total GPU time is still right after resumes.
    with (out_dir / "invocations.jsonl").open("a") as f:
        f.write(json.dumps({"started": started.isoformat(), "wall_s": time.perf_counter() - t_start,
                            "samples": len(todo), "per_task": args.per_task}) + "\n")

    preds = pd.read_json(pred_path, lines=True)
    preds = preds[preds["sample_id"].isin(ids)]
    invocations = pd.read_json(out_dir / "invocations.jsonl", lines=True)
    per_task = preds.groupby("task")["score"].mean().mul(100).round(2)
    summary = {
        "run": run,
        "config": vars(args),
        "press": repr(press),
        "dataset": {"repo": repo, "revision": revision, "config": str(args.context_length),
                    "split_file": f"splits/ruler{args.context_length}_s{args.seed}.json"},
        "n_samples": len(preds),
        "accuracy": round(preds["score"].mean() * 100, 2),
        "accuracy_per_task": per_task.to_dict(),
        "mean": preds[["context_tokens", "output_tokens", "kv_cache_tokens", "kv_cache_bytes", "prefill_s",
                       "ttft_s", "tpot_ms", "latency_s", "energy_j"]].mean().round(4).to_dict(),
        "peak_memory_gb": round(preds["peak_memory_bytes"].max() / 1e9, 2),
        "kv_bytes_per_token_uncompressed": kv_bytes_per_token(pipe.model.config),
        # Preempted invocations never log their wall time, so gpu_hours_logged is a lower bound;
        # compute_hours (sum of per-example latency) survives preemptions. GCP billing has the VM total.
        "gpu_hours_logged": round(invocations["wall_s"].sum() / 3600, 3),
        "compute_hours": round(preds["latency_s"].sum() / 3600, 3),
        "env": env_info(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(per_task.to_string())
    print(f"Overall accuracy: {summary['accuracy']}%  ->  {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
