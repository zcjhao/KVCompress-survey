#!/usr/bin/env bash
# Run every experiment for one compression method and ratio on the VM, then power the VM off.
# Run inside tmux from the project root, after setup_env.sh and `hf auth login`:
#   PRESS=snapkv RATIO=0.5 bash run_method.sh
#   PRESS=snapkv RATIO=0.5 NO_SHUTDOWN=1 bash run_method.sh   # keep the VM running afterwards
#   PRESS=none RATIO=0 bash run_method.sh                     # the uncompressed baseline (in baselines/)
#
# PRESS is the name of a file in scripts/presses/, or "none". All runs resume where they left off, so after a
# Spot preemption: start the VM and run the same command again (or let infra/supervise.sh do it with
# RUN="PRESS=snapkv RATIO=0.5 bash run_method.sh").
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p results

: "${PRESS:?set PRESS, e.g. PRESS=snapkv RATIO=0.5 bash run_method.sh}"
: "${RATIO:?set RATIO, e.g. PRESS=snapkv RATIO=0.5 bash run_method.sh}"
# Spell the ratio the way the scripts name their run folders (0.5, not .5 or 0.50).
RATIO=$(python -c "import sys; print(float(sys.argv[1]))" "$RATIO")
TAG="${PRESS}_r${RATIO}"

# Check the press loads before the power-off trap is set, so a typo doesn't shut the VM down.
(cd scripts && python -c "import sys; from common import build_press; build_press(sys.argv[1], float(sys.argv[2]))" "$PRESS" "$RATIO")

if [[ "${NO_SHUTDOWN:-0}" != "1" ]]; then
  # Power off even if a step fails, so a crashed run doesn't keep billing.
  trap 'echo "Powering off"; sudo poweroff' EXIT
fi

exec > >(tee -a "results/run_${TAG}.log") 2>&1
echo "=== ${TAG} started $(date -u) ==="

cd scripts
FLAGS=(--press "$PRESS" --compression-ratio "$RATIO")
# Smoke test: 1 example per task. These are the first examples of the full run below, so they are reused.
python eval_ruler.py --context-length 32768 --per-task 1 "${FLAGS[@]}"
# Skipped if already done, so a restart after preemption doesn't repeat it.
[[ -f "../results/latency_Llama-3.1-8B-Instruct_${TAG}/summary.json" ]] \
  || python bench_latency.py --lengths 4096 8192 16384 32768 65536 131072 "${FLAGS[@]}"
# Same --per-task as the baseline, so every method scores the baseline's examples.
python eval_ruler.py --context-length 32768 --per-task 50 "${FLAGS[@]}"
python eval_ruler.py --context-length 65536 --per-task 50 "${FLAGS[@]}"
python eval_ruler.py --context-length 131072 --per-task 20 "${FLAGS[@]}"

echo "=== ${TAG} finished $(date -u) ==="
