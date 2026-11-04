# Native DuoAttention — Method Report and Results Analysis

**Project:** COMS 6998 Scaling LLM, Fall 2026  
**Method:** DuoAttention-style retrieval/streaming heads with physically compacted KV caches  
**Model:** `meta-llama/Llama-3.1-8B-Instruct`  
**Experimental platform:** RunPod NVIDIA A100-SXM4-80GB

## 1. Method overview

Our implementation partitions the model's KV-head groups into **retrieval heads** and **streaming heads** using a frozen, pretrained DuoAttention head pattern. Retrieval heads keep the complete historical KV cache. Streaming heads maintain a compact cache consisting of the **first 128 sink tokens** and the **most recent 256 tokens**. During chunked prefill and autoregressive decoding, dropped middle-history KV entries are physically removed from the streaming cache rather than retained and masked.

The implementation is an independent functional PyTorch/Transformers adaptation, located in `native_methods/duoattention/`. Both the compressed model (`--mode native`) and its full-cache control (`--mode full`) run through the **same custom inference backend**. This pairing isolates the difference between the two KV-retention policies within the measured implementation.

The frozen head pattern is stored in `assets/duoattention_pattern/`; `configs/duoattention.json` supplies its location and checksum through `scripts/presses/duoattention_utils.py`. The source pattern contains 32 layers × 8 KV-head groups (256 groups total). With the target 50% KV reduction, the selected groups were:

| Context | Retrieval groups | Streaming groups | Sink / recent window |
|---|---:|---:|---|
| 32K | 126 | 130 | 128 / 256 tokens |
| 64K | 127 | 129 | 128 / 256 tokens |
| 128K | 127 | 129 | 128 / 256 tokens |

## 2. Experimental setup

| Setting | Value |
|---|---|
| GPU | 1 × NVIDIA A100-SXM4-80GB (RunPod) |
| Precision | BF16 |
| Python / PyTorch | 3.11.18 / `2.14.1+cu130` |
| CUDA runtime / Transformers / kvpress | 13.0 / 5.2.0 / 0.5.5 |
| Dataset | `Jhc6880/ruler-long`, pinned revision `deda02ec4c98e4cd6197f4738b03a1070767abf1` |
| RULER split | Team's committed seed-0 splits, 13 tasks |
| RULER examples | 50/task at 32K and 64K; 20/task at 128K |
| Prefill chunk length | 512 tokens |
| KV compression target | `r = 0.5` |
| Performance protocol | Batch 1; 128 fixed decode steps; 3 repeats per context and mode |
| CUDA allocator for long-context runs | `PYTORCH_ALLOC_CONF=expandable_segments:True` |

**Full** always retains the complete KV history, even when the paired run configuration includes `--compression-ratio 0.5`. **Native** uses the budget-selected retrieval/streaming partition. The 32K accuracy results below use the final Flash Attention causal-bias fix, consistent with the 64K and 128K runs.

## 3. RULER accuracy

### 3.1 Overall results

| Context length | Examples per mode | Full attention | Native DuoAttention (r=0.5) | Accuracy difference |
|---|---:|---:|---:|---:|
| 32K (32,768) | 650 | 88.68% | 85.19% | −3.49 pp |
| 64K (65,536) | 650 | 86.62% | 81.24% | −5.38 pp |
| 128K (131,072) | 260 | 77.69% | 67.69% | −10.00 pp |

At the measured 50% KV budget, the overall accuracy difference grows with context length. The 32K Native score is close to its Full control; by 128K, retaining long-range information becomes more consequential for several RULER tasks.

### 3.2 Task-level observations

Selected per-task scores, read directly from each run's `summary.json`:

| Context | Task | Full | Native | Change |
|---|---|---:|---:|---:|
| 32K | CWE | 27.80% | 0.40% | −27.40 pp |
| 32K | VT | 99.20% | 86.40% | −12.80 pp |
| 64K | VT | 96.80% | 67.20% | −29.60 pp |
| 64K | NIAH multi-value | 97.50% | 90.00% | −7.50 pp |
| 64K | QA 1 | 90.00% | 78.00% | −12.00 pp |
| 128K | NIAH multi-key 2 | 90.00% | 70.00% | −20.00 pp |
| 128K | NIAH multi-key 3 | 70.00% | 30.00% | −40.00 pp |
| 128K | NIAH multi-value | 96.25% | 82.50% | −13.75 pp |
| 128K | VT | 72.00% | 42.00% | −30.00 pp |

The three `niah_single_*` tasks all score **100%** in both modes at 32K and 64K. At 128K, Native scores **100% / 95% / 95%** on those tasks, versus **100% / 100% / 100%** for Full. This suggests that single-item retrieval is comparatively resilient under the tested head partition.

Multi-key retrieval and variable tracking are more sensitive at longer context. In particular, 128K `niah_multikey_3` drops by 40 pp and `vt` by 30 pp. A plausible interpretation is that longer-range or distributed evidence can become unavailable in the streaming groups, even while retrieval groups preserve full context. The results also show nonuniform effects: 128K `qa_2` is **55% Native versus 40% Full**, and both modes score **0% on 128K CWE**.

## 4. Physical KV and peak GPU allocated memory

The paired performance benchmark consists of **18 trials**: three context lengths × two modes × three repeats, using identical synthetic input tokens and fixed continuation token IDs for both modes. The following numbers are derived from `methods/duoattention_r0.5/performance/medians.csv` and `paired.csv`.

| Context | Full KV after prefill | Native KV after prefill | Physical KV saving | Peak PyTorch allocated: Full → Native | Peak allocated saving |
|---|---:|---:|---:|---:|---:|
| 32K | 4.295 GB | 2.139 GB | 50.19% | 21.068 → 18.912 GB | 10.23% |
| 64K | 8.590 GB | 4.287 GB | 50.10% | 26.034 → 21.731 GB | 16.53% |
| 128K | 17.180 GB | 8.548 GB | 50.24% | 35.967 → 27.335 GB | 24.00% |

The physical KV saving remains close to the target 50% at all three context lengths. The percentage reduction in **overall peak allocated memory** rises with context length, because the KV cache becomes a larger component of inference memory. Here **GB means decimal gigabytes** (10^9 bytes), and peak memory refers to PyTorch's measured peak allocations.

## 5. TTFT and decode performance

| Context | TTFT: Full → Native | TTFT speedup | Time per output token: Full → Native | Decode speedup |
|---|---:|---:|---:|---:|
| 32K | 7.658 → 5.978 s | 1.281× | 81.19 → 67.45 ms | 1.204× |
| 64K | 22.624 → 14.826 s | 1.526× | 142.65 → 93.73 ms | 1.522× |
| 128K | 75.647 → 41.936 s | 1.804× | 254.94 → 147.98 ms | 1.722× |

Both TTFT and decoding improve in the paired single-request benchmark, with the largest relative gains at 128K. The protocol times synchronized chunked prefill and first-token selection for TTFT, and averages 128 synchronized decode forwards for per-token latency. These results describe **same-backend single-request inference** on the RunPod A100, rather than a comparison of independent implementations or GPU providers.

## 6. Key findings

1. **Physical compression is achieved:** at a nominal 50% budget, Native stores roughly half of the Full control's KV tensor bytes at 32K, 64K, and 128K.
2. **Accuracy changes depend on context and task:** overall accuracy differences are −3.49, −5.38, and −10.00 pp as context grows; simple needle retrieval is relatively robust, while some multi-key and tracking tasks show larger decreases.
3. **Total GPU memory benefits grow with context:** peak allocated memory is reduced by 10.23%, 16.53%, and 24.00% as KV occupies more of the working memory.
4. **The paired performance benefit grows with sequence length:** decode speedups increase from 1.204× at 32K to 1.722× at 128K, alongside larger TTFT improvements.

Taken together, these results quantify an **accuracy–KV storage–latency trade-off** for the team's functional DuoAttention implementation at a 50% KV budget.

## 7. Reproduction and archived artifacts

Run from the repository root in the locked project environment, with access to the Llama model:

```bash
source .venv/bin/activate
export HF_HOME=/path/to/persistent/hf
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -m native_methods.duoattention.preflight
python -m unittest discover -s tests/native_duoattention -v

python -m native_methods.duoattention.runner --mode full \
  --context-length 32768 --per-task 50 --compression-ratio 0.5 \
  --chunk-size 512 --out /path/to/new-ruler-results
python -m native_methods.duoattention.runner --mode native \
  --context-length 32768 --per-task 50 --compression-ratio 0.5 \
  --chunk-size 512 --out /path/to/new-ruler-results

python -m native_methods.duoattention.bench_performance \
  --mode both --lengths 32768 65536 131072 \
  --repeats 3 --decode-tokens 128 --chunk-size 512 \
  --out /path/to/new-performance-results
```

For RULER at 64K use `--context-length 65536 --per-task 50`; for 128K use `--context-length 131072 --per-task 20`. Output paths above are new destinations to preserve the archived results.

**Results directory:** `methods/duoattention_r0.5/`

- `ruler32768_*`, `ruler65536_*`, `ruler131072_*`: Native results with `predictions.jsonl`, `summary.json`, `run_config.json`, and `invocations.jsonl`.
- `full_control/`: paired Full Attention RULER results at each context length.
- `performance/`: the 18 raw benchmark trials, mode medians, paired comparisons, and configuration.
- `code_sha256.txt`: archived source fingerprints; the recorded benchmark hashes match the submitted core implementation files.

See [`methods/duoattention_r0.5/README.md`](../methods/duoattention_r0.5/README.md) for the compact result summary, [`native_methods/duoattention/README.md`](../native_methods/duoattention/README.md) for implementation details, and the [original DuoAttention project](https://github.com/mit-han-lab/duo-attention) for the source method and head pattern.
