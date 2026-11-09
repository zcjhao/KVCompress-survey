# KV-cache quantization: FP8, KIVI-4, KIVI-2

Llama-3.1-8B-Instruct, RULER (13 tasks) at 32K, 64K and 128K, on the same examples as the
[baseline](../baselines/README.md): 50 per task at 32K and 64K (650 examples), 20 per task at 128K
(260). Runs are in `methods/fp8_r0.5`, `methods/kivi_r0.75` and `methods/kivi_r0.875`, each with
`summary.json` and per-example `predictions.jsonl`.

## Methods

Quantization keeps every token and stores each one in fewer bits. Eviction methods like SnapKV
and DuoAttention instead drop tokens or heads. The code is in `scripts/kv_quant.py`, `scripts/presses/fp8.py` and
`scripts/presses/kivi.py`.

| Method | Run folder | Storage | KV size vs BF16 (measured) |
|---|---|---|---|
| FP8 | `fp8_r0.5` | E4M3, scale fixed at 1.0 (as vLLM `kv_cache_dtype="fp8"` without calibration) | 50% |
| KIVI-4 | `kivi_r0.75` | int4: keys quantized per channel, values per token, group size 32, last 128 tokens kept in BF16 | 31% |
| KIVI-2 | `kivi_r0.875` | Same as KIVI-4 at int2 | 19% |

The run folder names use the nominal saving (4 bits = 0.75), so they line up with eviction ratios.
The measured sizes are larger than nominal because they include the per-group scales and zero
points and the BF16 residual. `cache_bytes()` counts all of these, so the KV sizes below are the
real stored bytes.

## Accuracy

Overall RULER accuracy (%) and the KV cache after prefill:

| | 32K | 64K | 128K | KV at 128K |
|---|---|---|---|---|
| Baseline (BF16) | 88.7 | 86.7 | 77.3 | 17.0 GB |
| FP8 | 89.2 | 86.3 | 76.8 | 8.5 GB |
| KIVI-4 | 89.2 | 86.5 | 78.0 | 5.3 GB |
| KIVI-2 | 86.1 | 80.4 | **64.2** | 3.2 GB |

**FP8 and KIVI-4 lose nothing measurable at any length.** Every difference from the baseline is
within about one standard error: roughly ±1.3 points overall at 32K and 64K and ±2.6 points at 128K.

**KIVI-2 degrades, and more as the context grows:** −2.6, −6.3 and −13.1 points.

### Where KIVI-2 fails

Per task (%), baseline / KIVI-2. Only the tasks that change are shown:

| Task | 32K | 64K | 128K |
|---|---|---|---|
| niah_multikey_2 | 100 / 94 | 96 / 94 | 95 / **35** |
| niah_multikey_3 | 98 / 82 | 100 / **56** | 70 / **5** |
| niah_multiquery | 98 / 97.5 | 98.5 / 98.5 | 100 / 91 |
| vt | 99.2 / 94.8 | 96.8 / 88.4 | 70 / 66 |
| qa_1 | 82 / 80 | 90 / 72 | 75 / 80 |

The loss is concentrated in multi-key needle retrieval. In niah_multikey_3, the haystack is full
of UUID keys that look alike, and the model must return the UUID value paired with one of them.

- **It is not noise.** At 128K, a per-task score has a standard error of about 10 points (20
  examples). The drop on niah_multikey_3 is 14/20 → 1/20 on the same examples (Fisher exact
  p ≈ 2×10⁻⁵), and on niah_multikey_2 it is 19/20 → 7/20 (p ≈ 7×10⁻⁵). The drop also grows steadily with
  length, from 98 to 82 to 56 to 5 for niah_multikey_3.
- **The failures are near misses.** KIVI-2 still answers with a UUID, but often the wrong one or a
  corrupted copy of the right one. For example, `4f7dae28-f549-…` instead of `4f7dae28-f271-…`, and
  `5c9f84ce-33e7-448b-…` instead of `5c9f85c3-33e7-4486-…`. Some answers are invented
  (`4e4a2a5c-…-4a1a3a3a3a3a`). At 2 bits, each key channel has only 4 levels, which is apparently too coarse to
  tell apart many near-identical keys or to copy a 36-character string exactly.
- **Tasks that need only one fact survive.** niah_single_1, the QA tasks and fwe barely move.

This task is also the hardest one for the eviction methods. At 128K, niah_multikey_3 is 10%
for SnapKV r=0.5 and 30% for DuoAttention r=0.5 (from those methods' own branches; see below).

## Speed, memory and energy

Medians on RULER. TTFT is time to first token and TPOT is time per output token. Peak memory is the
GPU peak, including the 16 GB of weights. Energy is GPU energy per example.

| 128K | TTFT | TPOT | Peak memory | Energy / example |
|---|---|---|---|---|
| Baseline | 33.3 s | 46.9 ms | 48.9 GB | 13.3 kJ |
| FP8 | 33.5 s | 80.1 ms | 40.3 GB | 13.8 kJ |
| KIVI-4 | 34.1 s | 245.7 ms | 37.1 GB | 15.2 kJ |
| KIVI-2 | 34.1 s | 243.0 ms | 34.9 GB | 15.5 kJ |

Latency sweep (random tokens, 128 decode tokens, median of 3), TPOT in ms:

| Context | Baseline | FP8 | KIVI-4 | KIVI-2 |
|---|---|---|---|---|
| 4K | 34.0 | 26.1 | 30.5 | 31.0 |
| 16K | 33.5 | 26.1 | 46.4 | 46.1 |
| 32K | 33.7 | 32.7 | 75.5 | 75.0 |
| 64K | 33.6 | 48.6 | 132.4 | 131.5 |
| 128K | 47.5 | 81.1 | 246.8 | 244.3 |

- **Memory saving is real.** At 128K, peak memory drops by 8.6 GB with FP8, 11.8 GB with KIVI-4 and
  14.0 GB with KIVI-2.
- **Decoding is slower, and this is a limitation of our implementation, not of quantization.** For each
  new token, our cache dequantizes the whole history back to BF16 in PyTorch, then runs standard
  attention. That cost grows with context length, so KIVI decodes 5× slower than the baseline at
  128K. KIVI's paper uses fused CUDA kernels that compute attention directly on the packed values,
  which makes decoding faster than BF16. FP8 needs only one type conversion, so it costs less, and
  it is faster than the baseline up to 16K. Don't read these TPOT numbers as the methods' real
  speed. Compare memory and accuracy across methods, not decoding speed.
- **TTFT barely changes.** Prefill runs in BF16, and the cache is quantized afterwards.
- **Energy per example is up to 16% higher** for the same reason: decoding takes longer.

## Comparison with eviction at the same memory

Eviction numbers are taken from the SnapKV branch (`snapkv`) and the DuoAttention PR (#4). Both
use the same 260 examples at 128K.

| 128K | KV kept | RULER |
|---|---|---|
| KIVI-4 | 31% | 78.0 |
| FP8 | 50% | 76.8 |
| DuoAttention r=0.5 | 50% | 67.7 |
| KIVI-2 | 19% | 64.2 |
| SnapKV r=0.25 | 75% | 59.5 |
| SnapKV r=0.5 | 50% | 44.4 |

Quantization keeps more accuracy per byte. KIVI-4 matches the baseline at 31% of the cache, and
KIVI-2 at 19% scores about the same as DuoAttention at 50%. One caveat: SnapKV in kvpress compresses
before it sees the question (see the main README), which puts it at a disadvantage.

## Caveats

- **Hardware.** The runs used one A100-SXM4-80GB on RunPod, not the GCP VM the baseline used. The
  GPU model and the pinned software versions are the same; see `env` in each `summary.json`.
  Accuracy is directly comparable. Treat absolute latency and energy differences of a few percent with care.
- **128K has 20 examples per task.** Overall accuracy has a standard error of about 2.6 points, and
  each per-task score about 10 points. Only large or consistent effects, like the KIVI-2 drop on
  multi-key retrieval, should be read per task at 128K.
- **cwe** is near 0 for every method at 64K and 128K, as in the baseline.
- **Compute.** 18.6 GPU-hours spent on RULER examples across the nine runs (from `compute_hours`),
  plus the latency sweeps.
