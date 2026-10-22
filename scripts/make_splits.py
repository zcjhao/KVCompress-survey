"""Write splits/ruler<length>_s<seed>.json: a fixed shuffled order of sample ids for each RULER task.

eval_ruler.py takes the first --per-task ids of each task from this file, so smaller runs are
subsets of larger ones. Run once per context length and commit the files; don't regenerate them.

    python scripts/make_splits.py --context-length 32768 65536 131072
"""

import argparse
import json

import numpy as np
from datasets import load_dataset

from common import REPO_ROOT, RULER_SOURCES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context-length", type=int, nargs="+", required=True, choices=sorted(RULER_SOURCES))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for length in args.context_length:
        repo, revision = RULER_SOURCES[length]
        path = REPO_ROOT / "splits" / f"ruler{length}_s{args.seed}.json"
        if path.exists():
            raise SystemExit(f"{path} exists; delete it only if you really mean to change the examples")
        tasks = load_dataset(repo, str(length), split="test", revision=revision).select_columns(["task"])["task"]
        rng = np.random.default_rng(args.seed)
        order = {}
        for task in sorted(set(tasks)):
            ids = np.array([i for i, t in enumerate(tasks) if t == task])
            order[task] = rng.permutation(ids).tolist()
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"dataset": repo, "revision": revision, "config": str(length),
                                    "seed": args.seed, "order": order}, indent=1))
        print(f"{path}: {len(order)} tasks, {min(len(v) for v in order.values())} ids per task")


if __name__ == "__main__":
    main()
