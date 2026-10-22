# KV-cache compression for long context: experiment harness

SLLM Fall 2026 final project, Theme 6. Every method (and the uncompressed baseline) runs through
the same scripts, so a comparison differs in one thing only: the KV-cache press.

## Fixed setup (do not change between methods)

| | |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct`, BF16, SDPA attention |
| Library | `kvpress==0.5.5` (other versions pinned in `requirements.lock.txt`) |
| Accuracy benchmark | RULER (`simonjegou/ruler`, 13 tasks), greedy decoding, official string-match scoring |
| Speed benchmark | Batch size 1, random tokens, 128 decode tokens, median of 3 runs |
| Seed | 0 (fixed subsample of RULER per task) |

Baseline KV-cache size for this model: 32 layers × 8 KV heads × 128 dim × 2 (K, V) × 2 bytes
= 128 KiB per token. That's 0.5 GiB at 4K tokens and 16 GiB at 128K, close to the 16 GB of weights.

## 1. Create the VM (from your laptop)

```bash
bash infra/create_vm.sh            # Spot g2-standard-4: 1x L4 24 GB, us-east1-b
SPOT=0 bash infra/create_vm.sh     # on-demand instead (never preempted)
gcloud compute ssh kv-l4 --zone us-east1-b
```

Costs in us-east1 (GCP list prices, Oct 2026; Spot prices can change):

| | L4 GPU | 4 vCPU + 16 GB RAM | Total |
|---|---|---|---|
| On-demand | $0.56/h | $0.15/h | about $0.71/h |
| Spot | $0.34/h | $0.09/h | about $0.42/h |

The 100 GB disk adds about $10/month and is billed while the VM is stopped, so delete the VM
when you no longer need it.

## 2. Environment (on the VM)

```bash
git clone https://github.com/zcjhao/KVCompress-survey && cd KVCompress-survey
bash setup_env.sh              # the first person: writes requirements.lock.txt, commit it
bash setup_env.sh --locked     # everyone else
source .venv/bin/activate
hf auth login                  # accept the Llama 3.1 licence on Hugging Face first
```

## 3. Baseline runs

```bash
tmux new -s run                # keeps the run alive if SSH disconnects
bash run_baseline.sh           # smoke test, RULER 4K and 8K, latency sweep; then powers the VM off
```

`run_baseline.sh` lists the individual commands if you want to run one step at a time.
Results go to `results/<run>/summary.json`, along with GPU, driver and package versions for the
paper. If a Spot VM is preempted, start it again and re-run the script: every run resumes.

## Adding a method

Add one entry to `build_press()` in `scripts/common.py`, then run the same commands with
`--press <name> --compression-ratio <r>`.

Note: the kvpress pipeline compresses the context *before* it sees the question
(query-agnostic compression). SnapKV's paper chooses tokens using the question, so expect
lower SnapKV scores here than the paper reports. Mention this in the comparison.
