"""Accuracy on RULER through the kvpress pipeline.

The baseline is just press=none, so every compression method later runs through exactly this
code path: same model, prompts, decoding and scoring.

    python scripts/eval_ruler.py --context-length 4096 --per-task 50
    python scripts/eval_ruler.py --context-length 4096 --per-task 2   # quick smoke test

Each run appends one JSON line per sample to results/<run>/predictions.jsonl, so a run that is
interrupted (e.g. a Spot preemption) picks up where it stopped when re-launched with the same arguments.
"""

import argparse
import ast
import json
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline

import kvpress  # noqa: F401  (registers the "kv-press-text-generation" pipeline)
from common import DEFAULT_MODEL, build_press, env_info, kv_bytes_per_token, set_seed


def score(pred: str, answers: list[str], task: str) -> float:
    """RULER's official metrics: QA tasks need any one reference (string_match_part); the
    others need every reference to appear in the output (string_match_all)."""
    pred = pred.lower()
    hits = [a.lower() in pred for a in answers]
    if task.startswith("qa"):
        return float(any(hits))
    return sum(hits) / len(hits)


def load_ruler(context_length: int, per_task: int, seed: int) -> pd.DataFrame:
    df = load_dataset("simonjegou/ruler", str(context_length), split="test").to_pandas()
    df["sample_id"] = df.index
    if per_task > 0:
        # Fixed-seed subsample so every method is scored on the same examples.
        df = df.groupby("task").sample(n=per_task, random_state=seed)
    df["answer"] = df["answer"].apply(lambda a: ast.literal_eval(a) if isinstance(a, str) else list(a))
    return df.sort_values("sample_id")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--context-length", type=int, default=4096, choices=[4096, 8192, 16384])
    p.add_argument("--per-task", type=int, default=50, help="examples per RULER task (0 = all 500)")
    p.add_argument("--press", default="none")
    p.add_argument("--compression-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = p.parse_args()

    set_seed(args.seed)
    model_tag = args.model.split("/")[-1]
    run = f"ruler{args.context_length}_{model_tag}_{args.press}_r{args.compression_ratio}_n{args.per_task}_s{args.seed}"
    out_dir = Path(args.out) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        done = {json.loads(line)["sample_id"] for line in pred_path.open()}

    df = load_ruler(args.context_length, args.per_task, args.seed)
    todo = df[~df["sample_id"].isin(done)]
    print(f"{run}: {len(df)} samples, {len(done)} already done, {len(todo)} to run")

    pipe = pipeline(
        "kv-press-text-generation",
        model=args.model,
        device="cuda:0",
        dtype=torch.bfloat16,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    press = build_press(args.press, args.compression_ratio)
    torch.cuda.reset_peak_memory_stats()

    with pred_path.open("a") as f:
        for i, row in enumerate(todo.itertuples(), 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = pipe(
                row.context,
                question=row.question,
                answer_prefix=row.answer_prefix,
                press=press,
                max_new_tokens=int(row.max_new_tokens),
            )["answer"]
            torch.cuda.synchronize()
            rec = {
                "sample_id": int(row.sample_id),
                "task": row.task,
                "score": score(pred, row.answer, row.task),
                "latency_s": time.perf_counter() - t0,
                "pred": pred,
                "answer": row.answer,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")

    preds = pd.read_json(pred_path, lines=True)
    per_task = preds.groupby("task")["score"].mean().mul(100).round(1)
    summary = {
        "run": run,
        "config": vars(args),
        "n_samples": len(preds),
        "accuracy": round(preds["score"].mean() * 100, 2),
        "accuracy_per_task": per_task.to_dict(),
        "mean_latency_s": round(preds["latency_s"].mean(), 3),
        # Peak covers only the samples run in this invocation.
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "kv_bytes_per_token_uncompressed": kv_bytes_per_token(pipe.model.config),
        "env": env_info(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(per_task.to_string())
    print(f"Overall accuracy: {summary['accuracy']}%  ->  {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
