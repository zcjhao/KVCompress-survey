# KV-cache compression for long context: experiment harness

SLLM Fall 2026 final project, Theme 6. Every method (and the uncompressed baseline) runs through
the same scripts, so a comparison differs in one thing only: the KV-cache press.

## Fixed setup (do not change between methods)

| | |
|---|---|
| Hardware | 1x A100 80GB (GCP `a2-ultragpu-1g`). Compare speed and energy only between runs on this GPU |
| Model | `meta-llama/Llama-3.1-8B-Instruct`, BF16, SDPA attention |
| Library | `kvpress==0.5.5` (all versions pinned in `requirements.lock.txt`) |
| Accuracy benchmark | RULER, 13 tasks, greedy decoding, official string-match scoring |
| Data | 32K-128K: [`Jhc6880/ruler-long`](https://huggingface.co/datasets/Jhc6880/ruler-long); 4K-16K: `simonjegou/ruler`. Both pinned to a commit in `scripts/common.py` |
| Examples | `splits/ruler<length>_s0.json`: the first `--per-task` ids of each task. Baseline: 50/task at 32K and 64K, 20/task at 128K |
| Speed benchmark | Batch size 1, random tokens, 128 decode tokens, median of 3 runs |

Baseline KV-cache size for this model: 32 layers × 8 KV heads × 128 dim × 2 (K, V) × 2 bytes
= 128 KiB per token. That's 4 GiB at 32K tokens and 16 GiB at 128K, close to the 16 GB of weights.

## 1. Create the VM (from your laptop)

```bash
bash infra/create_vm.sh            # Spot A100 80GB, us-central1-a
gcloud compute ssh kv-a100 --zone us-central1-a
```

If creation fails with "stockout", retry with `ZONE=us-central1-c`. Our project only has Spot
A100 quota (1 GPU, us-central1). `GPU=l4` creates a cheap L4 for debugging at short lengths.

Costs (GCP list prices, Oct 2026; Spot prices can change):

| | GPU | CPU + RAM | Total |
|---|---|---|---|
| A100 80GB Spot | $2.36/h | $0.68/h | about $3.04/h |
| L4 Spot | $0.34/h | $0.09/h | about $0.42/h |

The 100 GB disk adds about $10/month and is billed while the VM is stopped, so delete the VM
when you no longer need it.

## 2. Environment (on the VM)

```bash
git clone https://github.com/zcjhao/KVCompress-survey && cd KVCompress-survey
bash setup_env.sh --locked     # installs exactly requirements.lock.txt
source .venv/bin/activate
hf auth login                  # accept the Llama 3.1 licence on Hugging Face first
```

## 3. Baseline runs

```bash
tmux new -s run                # keeps the run alive if SSH disconnects
bash run_baseline.sh           # smoke test, latency sweep, RULER 32K/64K/128K; then powers the VM off
```

`run_baseline.sh` lists the individual commands if you want to run one step at a time. If a Spot
VM is preempted, start it again and re-run the script: every run resumes where it stopped.

## What each run records

`results/<run>/predictions.jsonl`, one line per example:

- **Example:** sample id, task, score, model output, reference answers
- **Tokens:** context, question and output token counts
- **KV cache kept after compression:** in tokens and in bytes
- **Time:** prefill, TTFT, TPOT and total latency
- **Memory and energy:** peak GPU memory, GPU energy (J)

`results/<run>/summary.json`:

- **Accuracy:** overall and per task
- **Means** of the per-example metrics
- **Run details:** press settings, dataset commit and split file, GPU-hours (across resumes)
- **Environment:** GPU, driver and package versions

`results/latency_*/summary.json`, per context length:

- TTFT, TPOT, KV size and peak memory
- Prefill energy, and decode energy per token

## Adding a method

Add one entry to `build_press()` in `scripts/common.py`, then run the same commands with
`--press <name> --compression-ratio <r>`.

Note: the kvpress pipeline compresses the context *before* it sees the question
(query-agnostic compression). SnapKV's paper chooses tokens using the question, so expect
lower SnapKV scores here than the paper reports. Mention this in the comparison.

## Regenerating the long-context data

Only needed to extend or audit the dataset; the uploaded copy is the reference.
`data/generate_ruler.sh` runs NVIDIA's RULER (pinned commit) on CPU, and
`data/build_hf_dataset.py` converts and uploads it. Never regenerate `splits/`: that would change
which examples everyone scores.
