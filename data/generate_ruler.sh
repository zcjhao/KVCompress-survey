#!/usr/bin/env bash
# Generate RULER data at long context lengths with NVIDIA's official RULER scripts.
# CPU only; run once, then upload with build_hf_dataset.py so the whole team scores the same examples.
#
#   uv venv --python 3.11 .venv-datagen && source .venv-datagen/bin/activate
#   uv pip install -r data/requirements-datagen.txt
#   hf auth login                      # the Llama 3.1 tokenizer is gated
#   bash data/generate_ruler.sh /path/to/workdir
#
# Settings match the 4K-16K data in simonjegou/ruler (the kvpress RULER dataset): Llama 3.1 tokenizer,
# base template, RULER's default seed. Output: <workdir>/data/<length>/<task>/validation.jsonl
set -euo pipefail

WORKDIR="$(cd "${1:?usage: generate_ruler.sh WORKDIR}" && pwd)"
NUM_SAMPLES="${NUM_SAMPLES:-100}"    # per task; more than we evaluate so runs can be extended later
LENGTHS=(${LENGTHS:-32768 65536 131072})
JOBS="${JOBS:-8}"
RULER_COMMIT=c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a   # NVIDIA/RULER main, 2026-07-22
TOKENIZER=meta-llama/Meta-Llama-3.1-8B
TASKS=(${TASKS:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3
       niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2})

cd "$WORKDIR"
if [[ ! -d RULER ]]; then
  git clone -q https://github.com/NVIDIA/RULER
  git -C RULER checkout -q "$RULER_COMMIT"
fi

# Haystack essays and the SQuAD / HotpotQA documents used by the QA tasks.
cd RULER/scripts/data/synthetic/json

# english_words.json (used by cwe) is stored in Git LFS; a plain clone only has a pointer file.
if head -c 40 english_words.json | grep -q git-lfs; then
  curl -fsSL -o english_words.json \
    "https://media.githubusercontent.com/media/NVIDIA/RULER/$RULER_COMMIT/scripts/data/synthetic/json/english_words.json"
  echo "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca  english_words.json" | shasum -a 256 -c -
fi
[[ -f PaulGrahamEssays.json ]] || python download_paulgraham_essay.py
# Same files as RULER's download_qa_dataset.sh, fetched with curl (macOS has no wget).
[[ -f squad.json ]] || curl -fsSL -o squad.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
[[ -f hotpotqa.json ]] || curl -fsSL --retry 3 -o hotpotqa.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json \
  || curl -fsSL -o hotpotqa.json https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json
cd "$WORKDIR/RULER/scripts"

for len in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "$len $task"
  done
done | xargs -P "$JOBS" -L 1 bash -c '
  python data/prepare.py --save_dir "'"$WORKDIR"'/data/$0" --benchmark synthetic --task "$1" \
    --tokenizer_path '"$TOKENIZER"' --tokenizer_type hf --max_seq_length "$0" \
    --model_template_type base --num_samples '"$NUM_SAMPLES"' > "'"$WORKDIR"'/log_$0_$1.txt" 2>&1 \
    && echo "done $0 $1" || echo "FAILED $0 $1 (see log_$0_$1.txt)"'

# RULER's prepare.py exits 0 even when the task script fails, so check every output file.
missing=0
for len in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    f="$WORKDIR/data/$len/$task/validation.jsonl"
    if [[ ! -f "$f" ]] || [[ $(wc -l < "$f") -ne $NUM_SAMPLES ]]; then
      echo "MISSING or incomplete: $f (see log_${len}_${task}.txt)"; missing=1
    fi
  done
done
exit $missing
