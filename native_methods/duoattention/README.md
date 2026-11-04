# Native DuoAttention — Physical Dual KV Cache

**Status:** Implementation and RULER evaluation completed on **RunPod NVIDIA A100-SXM4-80GB**. The final results cover 32K, 64K, and 128K context lengths, plus an 18-trial paired performance benchmark.

This directory contains the team's **independent functional PyTorch implementation** of DuoAttention-style retrieval/streaming heads. It integrates with the project’s pinned Transformers 5.2.0 Llama model and keeps physically separate, compact KV tensors for retrieval and streaming attention.

**Reports:** [Method results and analysis](../../reports/duoattention-method-report.md) · [Archived experiment results](../../methods/duoattention_r0.5/README.md)

## How it works

The frozen 32 × 8 head pattern in [`assets/duoattention_pattern/`](../../assets/duoattention_pattern/) specifies the ranking used to assign head groups. The budget planner chooses a set of **retrieval heads** (complete historical KV) and **streaming heads** (128 sink tokens + 256 recent tokens). Streaming KV is physically compacted after each prefill chunk and during decoding.

- **Retrieval attention:** full historical KV with a Flash SDPA-compatible lower-right causal bias.
- **Streaming attention:** sink + recent KV with absolute-position causal masking.
- **Prefill:** chunked processing, with 512-token chunks in the final experiments.
- **Decode:** one-token steps using absolute `cache_position`; the custom cache owns the retained KV tensors.
- **Paired control:** `--mode full` uses the same custom attention adapter with all groups in retrieval mode, so its KV cache remains uncompressed.

At target `r = 0.5`, head assignment was **126 retrieval / 130 streaming groups at 32K**, and **127 / 129 groups at 64K and 128K**. The head pattern and runtime settings are loaded through [`configs/duoattention.json`](../../configs/duoattention.json), [`scripts/presses/duoattention_utils.py`](../../scripts/presses/duoattention_utils.py) and the assets directory.

## Experimental environment

| Component | Final experimental setting |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (RunPod) |
| Model | `meta-llama/Llama-3.1-8B-Instruct`, BF16 |
| Python | 3.11.18 |
| PyTorch / CUDA | `2.14.1+cu130` / CUDA runtime 13.0 |
| Transformers / kvpress | `5.2.0` / `0.5.5` |
| RULER | Pinned team seed-0 splits; 650/650/260 examples per mode |
| Chunk size | 512 tokens |
| Compression target | `--compression-ratio 0.5` |
| CUDA allocator | `PYTORCH_ALLOC_CONF=expandable_segments:True` |

## Final RULER accuracy

Full and Native results use the same custom runner, model, split files and scoring. The 32K result is from the final Flash Attention causal-bias-fix rerun.

| Context | Full attention | Native DuoAttention (r=0.5) | Difference |
|---|---:|---:|---:|
| 32K (650 examples/mode) | 88.68% | 85.19% | −3.49 pp |
| 64K (650 examples/mode) | 86.62% | 81.24% | −5.38 pp |
| 128K (260 examples/mode) | 77.69% | 67.69% | −10.00 pp |

## Paired performance benchmark

The benchmark runs both modes on the same RunPod A100 with batch size 1, identical synthetic prompt tokens, 128 fixed continuation steps, and three repetitions per context/mode (18 trials).

| Context | Physical KV saving | TTFT speedup | Decode speedup | Peak PyTorch allocated saving |
|---|---:|---:|---:|---:|
| 32K | 50.19% | 1.281× | 1.204× | 10.23% |
| 64K | 50.10% | 1.526× | 1.522× | 16.53% |
| 128K | 50.24% | 1.804× | 1.722× | 24.00% |

The physical KV percentage is calculated from retained KV tensor bytes. TTFT and decode measurements refer to the paired **single-request** PyTorch benchmark. Raw trials and medians are archived in [`methods/duoattention_r0.5/performance/`](../../methods/duoattention_r0.5/performance/).

## Code structure

| File | Purpose |
|---|---|
| `attention.py` | Custom Llama attention adapter, retrieval/streaming routing, and Flash SDPA causal-bias handling |
| `dual_cache.py` | Physical dual KV cache storage, sink and recent maintenance |
| `runner.py` | RULER evaluation in Full or Native mode, plus synthetic smoke runs |
| `bench_performance.py` | Paired same-backend timing and memory benchmark |
| `preflight.py` | Dependency versions, CUDA environment and frozen-pattern validation |
| `bootstrap.sh` | Optional initialization of the original DuoAttention source checkout |
| `smoke_official.py` | Optional smoke test of the separate upstream DuoAttention implementation |
| `tests/native_duoattention/` | Dual-cache, causal-bias and small Transformers integration tests |

## Running the implementation

Use the pinned project dependencies from `requirements.lock.txt` and a CUDA GPU environment with access to the Llama-3.1-8B-Instruct model. All commands below run from the **repository root**:

```bash
source .venv/bin/activate
export HF_HOME=/path/to/persistent/hf
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Environment / head-pattern verification and unit tests
python -m native_methods.duoattention.preflight
python -m unittest discover -s tests/native_duoattention -v

# Fast functional smoke tests
python -m native_methods.duoattention.runner --mode full --smoke --chunk-size 128
python -m native_methods.duoattention.runner --mode native --smoke --chunk-size 128

# Paired RULER 32K evaluation
python -m native_methods.duoattention.runner --mode full \
  --context-length 32768 --per-task 50 --compression-ratio 0.5 \
  --chunk-size 512 --out /path/to/new-ruler-results
python -m native_methods.duoattention.runner --mode native \
  --context-length 32768 --per-task 50 --compression-ratio 0.5 \
  --chunk-size 512 --out /path/to/new-ruler-results

# Paired 32K / 64K / 128K performance benchmark
python -m native_methods.duoattention.bench_performance \
  --mode both --lengths 32768 65536 131072 \
  --repeats 3 --decode-tokens 128 --chunk-size 512 \
  --out /path/to/new-performance-results
```

To evaluate RULER 64K, use `--context-length 65536 --per-task 50`; for 128K, use `--context-length 131072 --per-task 20`. The team's pinned dataset revision and committed `splits/` files are used by the runner.

## Source and results

- [Method analysis](../../reports/duoattention-method-report.md)
- [Final archived results and reproducibility summary](../../methods/duoattention_r0.5/README.md)
- [Frozen head-pattern attribution](../../assets/duoattention_pattern/README.md)
- [Original DuoAttention project](https://github.com/mit-han-lab/duo-attention)
