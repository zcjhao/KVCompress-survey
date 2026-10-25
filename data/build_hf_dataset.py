"""Convert generate_ruler.sh output to the simonjegou/ruler format and upload it to the Hugging Face Hub.

Adapted from kvpress (Apache-2.0): evaluation/benchmarks/ruler/create_huggingface_dataset.py.
Each RULER prompt is split into context / question / answer_prefix so kvpress can compress the
context alone, exactly as for the 4K-16K data.

    python data/build_hf_dataset.py --data-dir <workdir>/data --repo-id <user-or-org>/ruler-long
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset

# Source: RULER scripts/data/synthetic/constants.py
QUESTION_PATTERNS = {
    "niah": re.compile(r"What (?:is|are all) the special magic"),
    "vt": re.compile(r"Question: Find all variables that are assigned the value"),
    "cwe": re.compile(r"Question: What are the 10 most common words in the above list\?"),
    "fwe": re.compile(r"Question: Do not provide any explanation\."),
    "qa": re.compile(r"Answer the question based on the given documents\."),
}
ANSWER_PATTERNS = {
    "niah": re.compile(r"The special magic"),
    "vt": re.compile(r"Answer:"),
    "cwe": re.compile(r"Answer:"),
    "fwe": re.compile(r"Answer:"),
    "qa": re.compile(r"Answer:"),
}
MAX_NEW_TOKENS = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}


def get_dataframe(path: Path) -> pd.DataFrame:
    frames = []
    for task_path in sorted(path.glob("*/validation.jsonl")):  # sorted: row order (= sample_id) is fixed
        df = pd.read_json(task_path, lines=True)
        task = task_path.parent.name
        kind = task.split("_")[0]

        def split(text):
            idx = list(QUESTION_PATTERNS[kind].finditer(text))[-1].start()
            context, qa = text[:idx], text[idx:]
            idx = ANSWER_PATTERNS[kind].search(qa).start()
            return context, qa[:idx], qa[idx:]

        def split_new(text, prefix):
            # Newer RULER keeps the answer prefix out of `input`. Move its leading space onto the question
            # so the prompt matches the older format used for the 4K-16K data exactly.
            idx = list(QUESTION_PATTERNS[kind].finditer(text))[-1].start()
            stripped = prefix.lstrip()
            return text[:idx], text[idx:] + prefix[: len(prefix) - len(stripped)], stripped

        if "answer_prefix" in df:
            parts = [split_new(t, a) for t, a in zip(df["input"], df["answer_prefix"])]
            df = df.drop(columns="answer_prefix")
        else:
            parts = df["input"].apply(split)
        df["context"], df["question"], df["answer_prefix"] = zip(*parts)
        df["task"] = task
        df["max_new_tokens"] = MAX_NEW_TOKENS[kind]
        frames.append(df)
    df = pd.concat(frames)[["context", "question", "answer_prefix", "outputs", "task", "max_new_tokens"]]
    return df.rename(columns={"outputs": "answer"}).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True, help="the data/ folder written by generate_ruler.sh")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    for path in sorted(args.data_dir.iterdir(), key=lambda q: int(q.name)):
        df = get_dataframe(path)
        print(f"{path.name}: {len(df)} rows, tasks: {df['task'].value_counts().to_dict()}")
        Dataset.from_pandas(df).push_to_hub(args.repo_id, config_name=path.name, split="test", private=args.private)


if __name__ == "__main__":
    main()
