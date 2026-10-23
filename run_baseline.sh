#!/usr/bin/env bash
# Run every baseline experiment on the VM, then power the VM off so it stops billing.
# Run inside tmux from the project root, after setup_env.sh and `hf auth login`:
#   bash run_baseline.sh
#   NO_SHUTDOWN=1 bash run_baseline.sh   # keep the VM running afterwards
#
# All runs resume where they left off, so after a Spot preemption: start the VM and run this again.
# A stopped VM keeps its disk, so copy the results back to your laptop afterwards:
#   gcloud compute instances start kv-a100 --zone us-central1-a
#   gcloud compute scp --recurse kv-a100:~/KVCompress-survey/results ./results --zone us-central1-a
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p results

if [[ "${NO_SHUTDOWN:-0}" != "1" ]]; then
  # Power off even if a step fails, so a crashed run doesn't keep billing.
  trap 'echo "Powering off"; sudo poweroff' EXIT
fi

exec > >(tee -a results/run_baseline.log) 2>&1
echo "=== baseline started $(date -u) ==="

cd scripts
# Smoke test: 1 example per task. These are the first examples of the full run below, so they are reused.
python eval_ruler.py --context-length 32768 --per-task 1
# Latency sweep next: quick, and shows up front that 128K fits in memory.
# Skipped if already done, so a restart after preemption doesn't repeat it.
[[ -f ../results/latency_Llama-3.1-8B-Instruct_none_r0.0/summary.json ]] \
  || python bench_latency.py --lengths 4096 8192 16384 32768 65536 131072
python eval_ruler.py --context-length 32768 --per-task 50
python eval_ruler.py --context-length 65536 --per-task 50
python eval_ruler.py --context-length 131072 --per-task 20

echo "=== baseline finished $(date -u) ==="
