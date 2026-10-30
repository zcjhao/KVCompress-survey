# KV-cache compression for long context: experiment harness

SLLM Fall 2026 final project, Theme 6. Every method (and the uncompressed baseline) runs through
the same scripts, so a comparison differs in one thing only: the KV-cache press.

**The baseline is done:** results and caveats are in [`baselines/`](baselines/README.md). Don't rerun it;
compare your method against those numbers.

## Fixed setup (do not change between methods)

| | |
|---|---|
| Hardware | 1x A100 80GB (GCP `a2-ultragpu-1g`). Compare speed and energy only between runs on this GPU |
| Model | `meta-llama/Llama-3.1-8B-Instruct`, BF16, SDPA attention |
| Library | `kvpress==0.5.5` (all versions pinned in `requirements.lock.txt`) |
| Accuracy benchmark | RULER, 13 tasks, greedy decoding, official string-match scoring |
| Data | 32K, 64K and 128K: [`Jhc6880/ruler-long`](https://huggingface.co/datasets/Jhc6880/ruler-long), pinned to a commit in `scripts/common.py` |
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

## 3. Run experiments

```bash
tmux new -s run                              # keeps the run alive if SSH disconnects
PRESS=snapkv RATIO=0.5 bash run_method.sh    # smoke test, latency sweep, RULER 32K/64K/128K; then powers the VM off
```

`run_method.sh` lists the individual commands if you want to run one step at a time.
`PRESS=none RATIO=0` reproduces the baseline (already done; don't rerun it). For the full workflow,
follow [Benchmarking a method, step by step](#benchmarking-a-method-step-by-step).

Time on the A100, per method setting: 32K at 50/task about 50 min, 64K about 2 h, 128K at 20/task
about 2.5 h; about $3 per hour on Spot.

## Running on Spot

Spot VMs get preempted; in our baseline run that happened twice in 6 hours, roughly every 30-40
minutes at first. Nothing is lost: the VM stops with its disk intact, and every script resumes from
the last finished example. To resume automatically, run the supervisor on your laptop and leave the
laptop awake. `RUN` is the command it relaunches, the same one you started:

```bash
RUN="PRESS=snapkv RATIO=0.5 bash run_method.sh" nohup bash infra/supervise.sh > supervise.log 2>&1 &   # VM=..., ZONE=... to override
```

It checks the VM every 2 minutes. When the VM has been preempted, it restarts the VM (retrying if
Spot capacity is sold out) and relaunches the run in tmux. It exits when the VM stops for any other
reason, which normally means the script finished and powered it off.

By hand instead: `gcloud compute instances start kv-a100 --zone us-central1-a`, SSH in, and run the
same command again in tmux.

## 4. Collect results and delete the VM

```bash
gcloud compute instances start kv-a100 --zone us-central1-a     # if the script powered it off
gcloud compute ssh kv-a100 --zone us-central1-a --command "grep -a '=== \|Traceback' KVCompress-survey/results/run_*.log"
gcloud compute scp --recurse kv-a100:~/KVCompress-survey/results ./results --zone us-central1-a
gcloud compute instances delete kv-a100 --zone us-central1-a    # stops the disk charge too
```

`results/` is in `.gitignore`, so nothing there gets pushed: move the run folders you want to keep
into `baselines/` or `methods/` by hand, then commit them. Don't push to `main` directly: work on
your own branch off `main` (e.g. `git checkout -b snapkv main`), push it, and merge it through a
pull request.
For a method, commit both `summary.json` and `predictions.jsonl` under `methods/<press>_r<ratio>/`,
one folder per run as in `baselines/` (e.g. `methods/snapkv_r0.5/ruler32768_.../`). The latency run
writes only `summary.json`. For the paper's compute statement, use the VM's running time from
`gcloud compute operations list --filter="targetLink~kv-a100"`; `compute_hours` in each summary
counts only time spent on examples.

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

Add one file, `scripts/presses/<name>.py`, with a `build(compression_ratio)` function that returns
a kvpress press (example in `scripts/presses/__init__.py`), then run the same commands with
`--press <name> --compression-ratio <r>`. Each method has its own file, so branches for different
methods don't conflict.

To score the same examples as the baseline, use the same `--context-length` and `--per-task`
(50 at 32K and 64K, 20 at 128K). `eval_ruler.py` always loads `Jhc6880/ruler-long` at the commit
pinned in `scripts/common.py` and takes the first `--per-task` ids of each task from the committed
`splits/` file, so a smaller `--per-task` gives a subset of the baseline's examples. Don't change
the dataset revision or the split files; the script refuses to run if they don't match.

Note: the kvpress pipeline compresses the context *before* it sees the question
(query-agnostic compression). SnapKV's paper chooses tokens using the question, so expect
lower SnapKV scores here than the paper reports. Mention this in the comparison.

**Check the KV-size numbers for your method.** `cache_bytes()` in `scripts/common.py` only counts
the cache's regular key/value tensors, which is right for token eviction (e.g. SnapKV) but can be
wrong for other kinds of methods. Quantized caches keep most of their data elsewhere, and kvpress's
head-wise presses (e.g. DuoAttention) only mask entries rather than freeing them. Before a full run,
check in your smoke test that the reported KV size matches what your method should keep, and fix
the measurement or note the limitation in the write-up if it doesn't.

## Benchmarking a method, step by step

Example: SnapKV at compression ratio 0.5. The team has one A100, so runs happen one at a time.

1. **Branch (laptop):** `git checkout -b snapkv main`.
2. **Add the press:** create `scripts/presses/snapkv.py` (see [Adding a method](#adding-a-method)).
   Commit and push the branch so the VM can check it out.
3. **VM and environment:** sections 1 and 2, then `git checkout snapkv` on the VM.
4. **Smoke test (VM):**
   `cd scripts && python eval_ruler.py --context-length 32768 --per-task 1 --press snapkv --compression-ratio 0.5`.
   In `results/ruler32768_Llama-3.1-8B-Instruct_snapkv_r0.5_s0/predictions.jsonl`, `kv_cache_tokens`
   should be about half of `context_tokens`; if they are equal, the press isn't compressing. The full
   run reuses these examples.
5. **Full run (VM):** `tmux new -s run`, then `PRESS=snapkv RATIO=0.5 bash run_method.sh`. It runs
   the smoke test, latency sweep and RULER runs with your press, logs to `results/run_snapkv_r0.5.log`, and
   powers the VM off at the end. Expect roughly 6 h and $17 per ratio (the baseline took 5.7
   A100-hours). If you use the supervisor, pass the same command:
   `RUN="PRESS=snapkv RATIO=0.5 bash run_method.sh" nohup bash infra/supervise.sh > supervise.log 2>&1 &`.
6. **Collect (laptop):** section 4, but grep `run_snapkv_r0.5.log` for `=== snapkv_r0.5\|Traceback`.
   Note the VM's running time for the compute statement before deleting the VM.
7. **Commit (laptop):**
   ```bash
   mkdir -p methods/snapkv_r0.5
   mv results/ruler*_snapkv_r0.5_s0 results/latency_*_snapkv_r0.5 methods/snapkv_r0.5/
   git add scripts/presses/snapkv.py methods/snapkv_r0.5
   git commit -m "SnapKV r0.5 results" && git push -u origin snapkv
   ```
   Then open a pull request into `main`.
8. **Compare** against [`baselines/`](baselines/README.md), reading its caveats first (cwe near 0,
   noise at 128K, flat TPOT, TTFT mostly prefill).

For another ratio, repeat steps 5-7 with a different `RATIO` and `methods/snapkv_r<ratio>/`.

## Regenerating the long-context data

Only needed to extend or audit the dataset; the uploaded copy is the reference.
`data/generate_ruler.sh` runs NVIDIA's RULER (pinned commit) on CPU, and
`data/build_hf_dataset.py` converts and uploads it. Never regenerate `splits/`: that would change
which examples everyone scores.
