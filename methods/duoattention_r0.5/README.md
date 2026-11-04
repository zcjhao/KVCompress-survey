# Native DuoAttention (physical dual KV cache) — PR submission

## What is included

- `native_methods/duoattention/`: native physical dual-KV-cache adapter, 32K/64K/128K runner, preflight, and single-request performance benchmark.
- `tests/native_duoattention/`: functional tests plus the Flash causal-bias regression tests.
- `assets/duoattention_pattern/`: frozen retrieval/streaming head pattern and attribution/license.
- Optional mask-only diagnostic plugin (`scripts/presses/duoattention*.py`, `tests/duoattention/`, `configs/duoattention.json`, `docs/duoattention.md`) if present. This plugin is NOT the physical-cache implementation.
- `methods/duoattention_r0.5/ruler*/`: Native DuoAttention RULER results (`summary.json`, `predictions.jsonl`, `run_config.json`, `invocations.jsonl`).
- `methods/duoattention_r0.5/full_control/`: full-attention outputs from the SAME custom inference backend for paired accuracy comparison.
- `methods/duoattention_r0.5/performance/`: 18-trial single-request paired benchmark and summaries.

This is designed to overlay a fresh checkout of the team's `main` on a NEW feature branch. It does not contain pre-existing team baselines, model weights, Python environments, HF cache, or temporary experiment data. Review all files before committing.

## Final accuracy (Llama-3.1-8B-Instruct; RULER)

| Context | Full (same backend) | Native r0.5 | Loss | Samples per mode |
|---|---:|---:|---:|---:|
| 32K | 88.68% | 85.19% | -3.49 pp | 650 |
| 64K | 86.62% | 81.24% | -5.38 pp | 650 |
| 128K | 77.69% | 67.69% | -10.00 pp | 260 |

32K uses the final Flash Attention fix (not the obsolete pre-fix 32K results).

## Performance (RunPod A100 SXM 80GB, batch 1, chunk 512, 128 decode steps, 3 repeats)

| Context | KV byte saving | TTFT speedup | Decode speedup | Peak allocated saving |
|---|---:|---:|---:|---:|
| 32K | 50.19% | 1.281x | 1.204x | 10.23% |
| 64K | 50.10% | 1.526x | 1.522x | 16.53% |
| 128K | 50.24% | 1.804x | 1.722x | 24.00% |

Values above are concise references; `performance/paired.csv` and `trials.jsonl` are authoritative measurement files.

## Reproduce (on a GPU with sufficient VRAM)

From repository root with dependencies installed and Llama license granted:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=/path/to/persistent/hf
python -m unittest discover -s tests/native_duoattention -v
python -m native_methods.duoattention.preflight
python -m native_methods.duoattention.runner --mode full --context-length 32768 --per-task 50 --compression-ratio 0.5 --chunk-size 512 --out /path/to/results
python -m native_methods.duoattention.runner --mode native --context-length 32768 --per-task 50 --compression-ratio 0.5 --chunk-size 512 --out /path/to/results
python -m native_methods.duoattention.bench_performance --mode both --lengths 32768 65536 131072 --repeats 3 --decode-tokens 128 --chunk-size 512
```

For 64K use `65536`, `--per-task 50`; for 128K use `131072`, `--per-task 20`. Use matching project dataset revision and splits; avoid changing baseline harness files.

## Important experimental caveats

1. The native attention adapter is a functional PyTorch implementation of DuoAttention's physical dual-KV-cache mechanism, **not the official optimized MIT inference kernel**.
2. `methods/duoattention_r0.5/full_control/` uses the same custom backend as Native, not the original teammate baseline implementation. The teammate's GCP baseline remains unchanged.
3. Performance results are **single-request PyTorch** measurements on a RunPod GPU; not concurrent serving-engine throughput. Do not directly interpret speed numbers as matched cross-provider GPU results.
4. The native runner uses frozen heads, streaming sink + recent tokens, and a locally adapted Flash Attention retrieval path.
5. These files were gathered from an on-volume archive. The packager checks JSONL record counts and basic structure, but review for any sensitive or unlicensed content prior to PR.
