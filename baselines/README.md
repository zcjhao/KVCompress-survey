# Baseline: full (uncompressed) KV cache

`meta-llama/Llama-3.1-8B-Instruct`, BF16, SDPA attention, batch size 1, 1x A100-SXM4-80GB (GCP Spot),
torch 2.14.1+cu130, transformers 5.2.0, kvpress 0.5.5. Run on 2026-10-22/23 with `run_baseline.sh`.
Each RULER folder holds the `summary.json` and the per-example `predictions.jsonl` written by the
scripts (model output, references, score, tokens, KV size, timings, memory, energy for every example),
so results can be re-scored or re-plotted without rerunning.

## RULER accuracy and cost

| Context | Examples | Accuracy | KV cache | TTFT (median) | TPOT (median) | Peak memory | Energy / example | Compute hours |
|---|---|---|---|---|---|---|---|---|
| 32K | 650 (50/task) | 88.7% | 4.2 GB | 3.8 s | 37.8 ms | 24.3 GB | 1.6 kJ | 0.83 |
| 64K | 650 (50/task) | 86.7% | 8.5 GB | 10.5 s | 38.6 ms | 32.5 GB | 4.3 kJ | 2.06 |
| 128K | 260 (20/task) | 77.3% | 17.0 GB | 33.3 s | 46.9 ms | 48.9 GB | 13.3 kJ | 2.49 |

Per task (%):

| Task | 32K | 64K | 128K |
|---|---|---|---|
| niah_single_1 / 2 / 3 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| niah_multikey_1 / 2 / 3 | 98 / 100 / 98 | 100 / 96 / 100 | 95 / 95 / 70 |
| niah_multivalue | 100 | 97.5 | 93.8 |
| niah_multiquery | 98 | 98.5 | 100 |
| vt | 99.2 | 96.8 | 70 |
| fwe | 90.7 | 86.7 | 66.7 |
| cwe | 31 | 5.2 | 0 |
| qa_1 (SQuAD) | 82 | 90 | 75 |
| qa_2 (HotpotQA) | 56 | 56 | 40 |

## Latency sweep (random tokens, 128 decode tokens, median of 3)

| Context | TTFT | TPOT | KV cache | Peak memory | Prefill energy | Decode energy / token |
|---|---|---|---|---|---|---|
| 4K | 0.31 s | 34.0 ms | 0.54 GB | 17.1 GB | 110 J | 5.3 J |
| 8K | 0.65 s | 33.5 ms | 1.07 GB | 18.1 GB | 237 J | 5.7 J |
| 16K | 1.46 s | 33.5 ms | 2.15 GB | 20.2 GB | 582 J | 6.2 J |
| 32K | 3.63 s | 33.7 ms | 4.29 GB | 24.3 GB | 1,423 J | 7.2 J |
| 64K | 10.1 s | 33.6 ms | 8.59 GB | 32.5 GB | 3,999 J | 9.3 J |
| 128K | 32.3 s | 47.5 ms | 17.2 GB | 48.9 GB | 12,830 J | 14.7 J |

## Caveats for comparing methods

- **cwe** is near 0 even without compression at 64K and 128K, so it can't separate methods there.
  Report it separately instead of letting it pull every method's average down equally.
- **128K uses 20 examples per task**, so each per-task score has a standard error of about 10 points.
  Treat per-task differences under about 15 points at 128K as noise. At 32K and 64K (50/task) it is
  about 4-7 points.
- **TPOT is flat until 64K.** Hugging Face's batch-1 decode loop is limited by Python and kernel-launch
  overhead, not memory bandwidth, so KV compression barely changes TPOT here before 128K. Decode
  speedups from compression need a serving engine (vLLM) or larger batches.
- **TTFT is mostly prefill**, which compression doesn't shorten: the full context is prefilled before
  it is compressed.
- **The first example of each invocation is slower** (GPU warm-up). Use medians for latency.
- **Cost:** the run took 5.72 A100-hours of Spot time, about $17 at list prices, including two
  preemptions.
