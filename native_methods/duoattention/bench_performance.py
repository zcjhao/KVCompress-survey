"""Paired, same-engine performance benchmark for native DuoAttention.

Add-only: invokes the installed model and NativeDuoAttention adapter, never edits
team scripts or dataset/accuracy results. Both modes receive identical prompt
and teacher-forced continuation token IDs. A100 CUDA benchmarking only.

Example:
 PYTORCH_ALLOC_CONF=expandable_segments:True python -m \
    native_methods.duoattention.bench_performance \
    --lengths 32768 65536 131072 --mode both --decode-tokens 32 --repeats 3
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = '/workspace/kvcompress/experiment-archive/duoattention/performance'


def mib(n):
    return round(n / (1024 ** 2), 3)


def gb(n):
    return round(n / 1e9, 3)


def sha256_file(p):
    p = Path(p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def generate_inputs(vocab, length, decode_tokens, seed):
    # Generate on CPU once per (length, repetition), then reuse EXACT tensors
    # in full and native modes. Forced continuation avoids content-dependent
    # EOS termination and diverging generation lengths/identities.
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    prompt = torch.randint(0, vocab, (1, length), generator=generator, dtype=torch.long)
    continuation = torch.randint(0, vocab, (1, decode_tokens), generator=generator, dtype=torch.long)
    return prompt.to('cuda:0'), continuation.to('cuda:0')


@torch.inference_mode()
def run_once(model, mode, ratio, length, decode_tokens, chunk_size,
             prompt, continuation):
    from native_methods.duoattention.runner import make_mask
    from native_methods.duoattention.attention import NativeDuoAttention
    from common import kv_bytes_per_token

    mask, sink, recent, plan = make_mask(model.config, ratio, mode, length)
    adapter = NativeDuoAttention(model, mask, sink_size=sink, recent_size=recent)

    # Peak tracks model weights plus allocations made during this request,
    # NOT total physical device memory consumed by other applications.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()

    try:
        with adapter:
            adapter.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            last_output = None
            for start in range(0, length, chunk_size):
                stop = min(start + chunk_size, length)
                last_output = model(
                    input_ids=prompt[:, start:stop],
                    use_cache=False,
                    cache_position=torch.arange(start, stop, device=prompt.device),
                    logits_to_keep=1,
                )
            # TTFT: prefill all prompt tokens and calculate first generated
            # token via argmax (no extra question tokens, no network).
            first_id = last_output.logits[:, -1].argmax(dim=-1)
            torch.cuda.synchronize()
            ttft_s = time.perf_counter() - t0
            del last_output, first_id

            kv_after_prefill = adapter.cache.breakdown()
            peak_prefill_allocated = torch.cuda.max_memory_allocated()
            peak_prefill_reserved = torch.cuda.max_memory_reserved()

            # Fixed teacher-forced continuation: every mode sees precisely the
            # same token IDs and decodes exactly decode_tokens steps. This is
            # a controlled compute benchmark, not a text-quality benchmark.
            decode_times = []
            for i in range(decode_tokens):
                x = continuation[:, i:i + 1]
                torch.cuda.synchronize()
                d0 = time.perf_counter()
                output = model(
                    input_ids=x,
                    use_cache=False,
                    cache_position=torch.tensor([length + i], device=prompt.device),
                    logits_to_keep=1,
                )
                next_id = output.logits[:, -1].argmax(dim=-1)
                torch.cuda.synchronize()
                decode_times.append(time.perf_counter() - d0)
                del output, next_id

            peak_total_allocated = torch.cuda.max_memory_allocated()
            peak_total_reserved = torch.cuda.max_memory_reserved()
            kv_after_decode = adapter.cache.breakdown()
            theoretical_bytes = kv_bytes_per_token(model.config, dtype_bytes=2) * length
            decode_s = sum(decode_times)
            row = {
                'length': length,
                'mode': mode,
                'compression_ratio_setting': ratio,
                'retrieval_head_groups': int(mask.sum()),
                'streaming_head_groups': int(mask.size - mask.sum()),
                'sink_tokens': sink,
                'recent_tokens': recent,
                'batch_size': 1,
                'chunk_size': chunk_size,
                'decode_steps': decode_tokens,
                'ttft_s': ttft_s,
                'tpot_ms': 1000.0 * decode_s / decode_tokens,
                'decode_total_s': decode_s,
                'decode_tokens_per_s': decode_tokens / decode_s,
                'total_latency_s': ttft_s + decode_s,
                'generated_tokens_per_s_including_prefill': (1 + decode_tokens) / (ttft_s + decode_s),
                'prompt_tokens_per_s': length / ttft_s,
                'kv_cache_bytes_prefill': int(kv_after_prefill['kv_cache_bytes']),
                'kv_cache_bytes_after_decode': int(kv_after_decode['kv_cache_bytes']),
                'full_theoretical_kv_bytes_prefill': int(theoretical_bytes),
                'kv_saving_ratio': 1.0 - kv_after_prefill['kv_cache_bytes'] / theoretical_bytes,
                'retrieval_kv_bytes_prefill': int(kv_after_prefill['retrieval_kv_bytes']),
                'streaming_kv_bytes_prefill': int(kv_after_prefill['streaming_kv_bytes']),
                'peak_prefill_allocated_bytes': int(peak_prefill_allocated),
                'peak_prefill_reserved_bytes': int(peak_prefill_reserved),
                'peak_total_allocated_bytes': int(peak_total_allocated),
                'peak_total_reserved_bytes': int(peak_total_reserved),
                'model_loaded_allocated_bytes': int(baseline_allocated),
                'note': 'Single request, synthetic tokens, fixed continuation, no serving engine',
            }
        return row
    finally:
        # Exception-safe unpatch and release all request-scoped physical KV.
        adapter.reset()
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def median_summary(rows):
    grouped = {}
    metrics = [
        'ttft_s', 'tpot_ms', 'decode_tokens_per_s',
        'total_latency_s', 'generated_tokens_per_s_including_prefill',
        'prompt_tokens_per_s', 'kv_cache_bytes_prefill', 'kv_cache_bytes_after_decode',
        'kv_saving_ratio', 'peak_prefill_allocated_bytes', 'peak_prefill_reserved_bytes',
        'peak_total_allocated_bytes', 'peak_total_reserved_bytes',
        'model_loaded_allocated_bytes',
    ]
    for r in rows:
        key = (r['length'], r['mode'])
        grouped.setdefault(key, []).append(r)
    results = []
    for (length, mode), group in sorted(grouped.items()):
        item = {
            'length': length, 'mode': mode,
            'trials': len(group),
            'chunk_size': group[0]['chunk_size'],
            'decode_steps': group[0]['decode_steps'],
            'retrieval_head_groups': group[0]['retrieval_head_groups'],
            'streaming_head_groups': group[0]['streaming_head_groups'],
        }
        for name in metrics:
            item[name] = statistics.median(float(r[name]) for r in group)
        results.append(item)
    return results


def paired_summary(medians):
    by_length = {}
    for r in medians:
        by_length.setdefault(r['length'], {})[r['mode']] = r
    comparisons = []
    for length, modes in sorted(by_length.items()):
        if not ('full' in modes and 'native' in modes):
            continue
        full, native = modes['full'], modes['native']
        comparisons.append({
            'length': length,
            'full_ttft_s': full['ttft_s'],
            'native_ttft_s': native['ttft_s'],
            'ttft_speedup_full_over_native': full['ttft_s'] / native['ttft_s'],
            'full_tpot_ms': full['tpot_ms'],
            'native_tpot_ms': native['tpot_ms'],
            'decode_speedup_native_over_full': full['tpot_ms'] / native['tpot_ms'],
            'full_kv_cache_bytes': full['kv_cache_bytes_prefill'],
            'native_kv_cache_bytes': native['kv_cache_bytes_prefill'],
            'physical_kv_saving_ratio': 1 - native['kv_cache_bytes_prefill'] / full['kv_cache_bytes_prefill'],
            'full_peak_allocated_bytes': full['peak_total_allocated_bytes'],
            'native_peak_allocated_bytes': native['peak_total_allocated_bytes'],
            'peak_allocated_saving_ratio': 1 - native['peak_total_allocated_bytes'] / full['peak_total_allocated_bytes'],
            'full_peak_reserved_bytes': full['peak_total_reserved_bytes'],
            'native_peak_reserved_bytes': native['peak_total_reserved_bytes'],
        })
    return comparisons


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['full','native','both'], default='both')
    p.add_argument('--lengths', nargs='+', type=int, default=[32768, 65536, 131072])
    p.add_argument('--compression-ratio', type=float, default=0.5)
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--decode-tokens', type=int, default=32,
                   help='Fixed teacher-forced continuation steps, in addition to first token')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup-length', type=int, default=512)
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--out', default=DEFAULT_OUT)
    p.add_argument('--dry-run', action='store_true', help='Print run plan, without touching GPU or files')
    args = p.parse_args(argv)
    if (any(x < 1 for x in args.lengths) or args.repeats < 1 or args.decode_tokens < 1
        or args.chunk_size < 1 or args.warmup_length < 1 or not 0 <= args.compression_ratio < 1):
        p.error('positive lengths/repeats/decode/chunk/warmup and compression-ratio in [0,1) required')
    args.lengths = list(dict.fromkeys(args.lengths))
    modes = ['full','native'] if args.mode == 'both' else [args.mode]
    plan = {'modes': modes, 'lengths': args.lengths, 'repeats': args.repeats,
            'decode_steps': args.decode_tokens, 'chunk_size': args.chunk_size,
            'comparison': 'identical prompt tokens and identical forced decode tokens',
            'expected_trials': len(modes) * len(args.lengths) * args.repeats,
            'out': args.out}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    if not torch.cuda.is_available():
        raise SystemExit('CUDA not available: must run on the RunPod A100')
    if torch.cuda.device_count() != 1:
        print('WARNING: multiple GPUs visible; only cuda:0 will be used', file=sys.stderr)
    from native_methods.duoattention.runner import model_and_pipe, make_mask
    from native_methods.duoattention.attention import NativeDuoAttention
    from common import env_info, DEFAULT_MODEL

    torch.manual_seed(args.seed)
    pipe = model_and_pipe(DEFAULT_MODEL)
    model = pipe.model
    gpu = torch.cuda.get_device_name(0)
    info = {
        'args': vars(args), 'plan': plan, 'gpu': gpu,
        'env': env_info(), 'python': platform.python_version(),
        'allocator_config': os.environ.get('PYTORCH_ALLOC_CONF'),
        'start_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in (
            ROOT / 'native_methods/duoattention/attention.py',
            ROOT / 'native_methods/duoattention/dual_cache.py',
            ROOT / 'native_methods/duoattention/runner.py',
            ROOT / 'native_methods/duoattention/bench_performance.py',
        )},
        'limits': [
            'One concurrent request, batch=1; NOT a production serving-engine throughput test',
            'Forced continuation ensures identical decode inputs; it is not a generation quality test',
            'Per-repetition medians; input token IDs are synthetic but identical between modes',
            'TTFT = entire chunked prefill + first-token argmax; excludes model loading and tokenizer',
            'TPOT = mean synchronized one-token forwards with argmax after TTFT',
            'GPU peak allocated/reserved measures torch allocator for THIS process, not full nvidia-smi',
        ]
    }
    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out_dir = Path(args.out) / f'bench_{now}'
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / 'config.json').write_text(json.dumps(info, indent=2))
    print('A100 benchmark output:', out_dir, flush=True)

    print('Warmup:', modes, 'length', args.warmup_length, flush=True)
    for mode in modes:
        warm_prompt, warm_decode = generate_inputs(model.config.vocab_size,
                                                  args.warmup_length, 4,
                                                  args.seed + 999)
        warmup = run_once(model, mode, args.compression_ratio, args.warmup_length,
                          4, min(args.chunk_size, args.warmup_length), warm_prompt, warm_decode)
        del warmup, warm_prompt, warm_decode

    rows = []
    for length in args.lengths:
        for rep in range(args.repeats):
            seed = args.seed + (length * 37) + rep
            prompt, continuation = generate_inputs(model.config.vocab_size, length,
                                                   args.decode_tokens, seed)
            order = modes if rep % 2 == 0 else list(reversed(modes))
            for mode in order:
                print(f'RUN length={length} repeat={rep+1}/{args.repeats} mode={mode}', flush=True)
                try:
                    row = run_once(model, mode, args.compression_ratio, length,
                                   args.decode_tokens, args.chunk_size, prompt,
                                   continuation)
                except Exception as exc:
                    err = {'length':length, 'repeat':rep+1, 'mode':mode,
                           'error_type':type(exc).__name__, 'error':str(exc)}
                    with (out_dir / 'errors.jsonl').open('a') as f:
                        f.write(json.dumps(err) + '\n')
                    print('Benchmark stopped; completed trials are safe in', out_dir, file=sys.stderr)
                    raise
                row['repeat'] = rep + 1
                row['input_seed'] = seed
                row['attention_sha256'] = info['source_sha256']['native_methods/duoattention/attention.py']
                rows.append(row)
                with (out_dir / 'trials.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f"  TTFT={row['ttft_s']:.3f}s TPOT={row['tpot_ms']:.2f}ms "
                      f"KV={gb(row['kv_cache_bytes_prefill']):.3f}GB "
                      f"peak_alloc={gb(row['peak_total_allocated_bytes']):.3f}GB "
                      f"peak_reserved={gb(row['peak_total_reserved_bytes']):.3f}GB", flush=True)
            del prompt, continuation

    medians = median_summary(rows)
    comparisons = paired_summary(medians)
    result = {'run': info, 'medians': medians, 'paired': comparisons,
              'end_utc': datetime.now(timezone.utc).isoformat()}
    (out_dir / 'summary.json').write_text(json.dumps(result, indent=2))
    write_csv(out_dir / 'medians.csv', medians)
    write_csv(out_dir / 'paired.csv', comparisons)
    print('\nFINAL PAIRED MEDIANS (speedup >1 means native is faster):', flush=True)
    for c in comparisons:
        print(f"{c['length']:6d}: KV saving={c['physical_kv_saving_ratio']:.2%}, "
              f"TTFT speedup={c['ttft_speedup_full_over_native']:.3f}x, "
              f"decode speedup={c['decode_speedup_native_over_full']:.3f}x, "
              f"peak-allocated saving={c['peak_allocated_saving_ratio']:.2%}", flush=True)
    print('SAVED:', out_dir, flush=True)


if __name__ == '__main__':
    main()
