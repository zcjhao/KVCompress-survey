#!/usr/bin/env bash
# Run every baseline experiment on the VM, then power the VM off so it stops billing.
# Run inside tmux from the project root, after setup_env.sh and `hf auth login`:
#   bash run_baseline.sh
#   NO_SHUTDOWN=1 bash run_baseline.sh   # keep the VM running afterwards
#
# All runs resume where they left off, so after a Spot preemption: start the VM and run this again.
# A stopped VM keeps its disk, so copy the results back to your laptop afterwards:
#   gcloud compute instances start kv-l4 --zone us-east1-b
#   gcloud compute scp --recurse kv-l4:~/KVCompress-survey/results ./results --zone us-east1-b
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
python eval_ruler.py --context-length 4096 --per-task 2    # smoke test: fails fast if something is broken
python eval_ruler.py --context-length 4096 --per-task 50
python eval_ruler.py --context-length 8192 --per-task 50
python bench_latency.py --lengths 1024 2048 4096 8192 16384

echo "=== baseline finished $(date -u) ==="
