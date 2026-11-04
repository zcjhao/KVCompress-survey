"""Method-specific RULER/short smoke runner for the team's locked environment.

No changes to team's eval_ruler.py, common.py, model weights, dataset splits,
or requirements.lock.txt. Only physical dual-KV-cache implementation is custom.
"""
from __future__ import annotations
import argparse
import ast
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline
import kvpress  # registers the same prompt-preprocessing pipeline as the team

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import DEFAULT_MODEL, RULER_SOURCES, env_info, kv_bytes_per_token, GpuEnergy, set_seed
from eval_ruler import score, load_split
from presses.duoattention_utils import read_pattern, binary_heads, budget_plan
from native_methods.duoattention.attention import NativeDuoAttention


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_forward(model, ids, start, *, logits_to_keep=1):
    synchronize()
    t0 = time.perf_counter()
    outputs = model(input_ids=ids, use_cache=False,
                    cache_position=torch.arange(start, start + ids.shape[1], device=ids.device),
                    logits_to_keep=logits_to_keep)
    synchronize()
    return outputs, time.perf_counter() - t0


@torch.inference_mode()
def infer(pipe, adapter, context, question, answer_prefix, max_new_tokens, chunk_size):
    """Identical kvpress prompt construction, but physically separated per-head caches.

    No kvpress press is applied. The native adapter owns all KV state.
    """
    tokenizer = pipe.tokenizer
    x = pipe.preprocess(context, questions=[question], answer_prefix=answer_prefix,
                        max_context_length=min(tokenizer.model_max_length, 10**9))
    ids = x['context_ids'].to(pipe.model.device)
    question_ids = x['questions_ids'][0].to(pipe.model.device)
    adapter.reset()
    torch.cuda.reset_peak_memory_stats()
    energy = GpuEnergy()
    e0 = energy.read()
    total_start = time.perf_counter()
    prefill_s = 0.0
    for start in range(0, ids.shape[1], chunk_size):
        _, elapsed = timed_forward(pipe.model, ids[:, start:start + chunk_size], start)
        prefill_s += elapsed
    context_breakdown = adapter.cache.breakdown()
    context_tokens = ids.shape[1]
    theoretical = kv_bytes_per_token(pipe.model.config, dtype_bytes=2) * context_tokens
    if max_new_tokens < 1 or question_ids.shape[1] == 0:
        raise ValueError('Question and max_new_tokens must be nonempty')
    outputs, question_s = timed_forward(pipe.model, question_ids, context_tokens)
    next_id = outputs.logits[0, -1].argmax().reshape(1, 1)
    generated = [next_id.item()]
    eos = pipe.model.generation_config.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    eos = set(eos or [])
    decode_durations = []
    next_pos = context_tokens + question_ids.shape[1]
    for _ in range(max_new_tokens - 1):
        if generated[-1] in eos:
            break
        outputs, elapsed = timed_forward(pipe.model, next_id, next_pos)
        decode_durations.append(elapsed)
        next_pos += 1
        next_id = outputs.logits[0, -1].argmax().reshape(1, 1)
        generated.append(next_id.item())
    synchronize()
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    e1 = energy.read()
    return {
        'pred': answer, 'context_tokens': context_tokens,
        'question_tokens': question_ids.shape[1], 'output_tokens': len(generated),
        'kv_cache_tokens': context_tokens,
        'kv_cache_bytes': context_breakdown['kv_cache_bytes'],
        'full_context_kv_bytes': theoretical,
        'physical_retained_ratio': context_breakdown['kv_cache_bytes'] / theoretical,
        'physical_compression_ratio': 1 - context_breakdown['kv_cache_bytes'] / theoretical,
        'retrieval_kv_bytes': context_breakdown['retrieval_kv_bytes'],
        'streaming_kv_bytes': context_breakdown['streaming_kv_bytes'],
        'prefill_s': prefill_s, 'ttft_s': prefill_s + question_s,
        'tpot_ms': sum(decode_durations) * 1e3 / max(len(decode_durations), 1),
        'latency_s': time.perf_counter() - total_start,
        'peak_memory_bytes': torch.cuda.max_memory_allocated(),
        'energy_j': None if e0 is None or e1 is None else e1 - e0,
    }


def make_mask(config, ratio, mode, context_length):
    scores, sink, recent = read_pattern()
    if tuple(scores.shape) != (config.num_hidden_layers, config.num_key_value_heads):
        raise ValueError('Official frozen pattern does not match the requested Llama model')
    if mode == 'full':
        return np.ones_like(scores, dtype=bool), sink, recent, None
    if not 0 <= ratio < 1:
        raise ValueError('compression_ratio must be in [0, 1)')
    # Use context length as a reference budget. For shorter final examples, preserve
    # this head selection; actual physical ratio is always recorded separately.
    target = max(1 - ratio, min(context_length, sink + recent) / context_length)
    plan = budget_plan(context_length, target, sink, recent, layers=config.num_hidden_layers,
                       kv_heads=config.num_key_value_heads,
                       dim=config.hidden_size // config.num_attention_heads)
    mask = binary_heads(scores, plan['streaming_heads']).astype(bool)
    return mask, sink, recent, plan


def model_and_pipe(name):
    # Same pipeline/model preprocessing as the team; no new package installs.
    p = pipeline('kv-press-text-generation', model=name, device='cuda:0',
                 dtype=torch.bfloat16, model_kwargs={'attn_implementation': 'sdpa'})
    p.model.eval()
    return p


def summary_json(out_dir, run, args, pred_path, invocations, press, model):
    preds = pd.read_json(pred_path, lines=True)
    selected = set(load_split(args.context_length, args.seed, args.per_task)[1])
    preds = preds[preds.sample_id.isin(selected)]
    per_task = preds.groupby('task')['score'].mean().mul(100).round(2)
    metric_cols = ['context_tokens', 'output_tokens', 'kv_cache_tokens','kv_cache_bytes',
                   'physical_retained_ratio', 'physical_compression_ratio',
                   'prefill_s','ttft_s','tpot_ms','latency_s','energy_j']
    summary = {
        'run': run, 'implementation': 'physical_dual_kv_cache_functional_port',
        'official_kernel_equivalence': False,
        'config': vars(args), 'press': press,
        'dataset': {'repo': RULER_SOURCES[args.context_length][0],
                    'revision': RULER_SOURCES[args.context_length][1],
                    'config': str(args.context_length),
                    'split_file': f'splits/ruler{args.context_length}_s{args.seed}.json'},
        'n_samples': len(preds),
        'accuracy': round(preds.score.mean() * 100, 2),
        'accuracy_per_task': per_task.to_dict(),
        'mean': preds[metric_cols].mean().round(6).to_dict(),
        'peak_memory_gb': round(preds.peak_memory_bytes.max() / 1e9, 3),
        'kv_bytes_per_token_uncompressed': kv_bytes_per_token(model.config),
        'compute_hours': round(preds.latency_s.sum() / 3600, 3),
        'gpu_hours_logged': round(sum(x['wall_s'] for x in invocations) / 3600, 3),
        'env': env_info(),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(per_task.to_string())
    print(f"Overall accuracy: {summary['accuracy']}% -> {out_dir / 'summary.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['full', 'native'], default='native',
                   help='full uses the SAME patched engine with all heads as retrieval controls')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--compression-ratio', type=float, default=0.5)
    p.add_argument('--context-length', type=int, default=32768, choices=sorted(RULER_SOURCES))
    p.add_argument('--per-task', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--out', default=str(ROOT / 'results'))
    p.add_argument('--smoke', action='store_true', help='One synthetic input; no dataset download')
    args = p.parse_args()
    if args.chunk_size <= 0 or args.per_task < 1:
        p.error('chunk-size and per-task must be positive')
    if not torch.cuda.is_available():
        p.error('This runner requires CUDA (run it in your RunPod A100 environment)')
    set_seed(args.seed)
    pipe = model_and_pipe(args.model)
    mask, sink, recent, plan = make_mask(pipe.model.config, args.compression_ratio,
                                         args.mode, args.context_length)
    adapter = NativeDuoAttention(pipe.model, mask, sink_size=sink, recent_size=recent)
    print('Mode:', args.mode, 'retrieval/streaming head groups:',
          int(mask.sum()), '/', mask.size - int(mask.sum()))
    if args.smoke:
        context = 'The verification keyword is bluebird. ' + ('A paragraph about testing. ' * 70)
        with adapter:
            rec = infer(pipe, adapter, context, '\nWhat is the verification keyword?', '', 16,
                        args.chunk_size)
        print(json.dumps(rec, indent=2))
        return
    split, ids = load_split(args.context_length, args.seed, args.per_task)
    repo, revision = RULER_SOURCES[args.context_length]
    if (split['dataset'], split['revision']) != (repo, revision):
        raise SystemExit('Pinned split and dataset revision do not match')
    tag = args.model.split('/')[-1]
    run = (f'ruler{args.context_length}_{tag}_duoattention_{args.mode}_r'
           f'{args.compression_ratio}_s{args.seed}_chunk{args.chunk_size}')
    out_dir = Path(args.out) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / 'predictions.jsonl'
    settings = {'mode': args.mode, 'model': args.model, 'ratio': args.compression_ratio,
                'chunk_size': args.chunk_size, 'context_length': args.context_length,
                'seed': args.seed, 'pattern': 'assets/duoattention_pattern',
                'data_repo': repo, 'data_revision': revision,
                'implementation': 'physical_dual_kv_cache_functional_port'}
    setting_path = out_dir / 'run_config.json'
    if setting_path.exists() and json.loads(setting_path.read_text()) != settings:
        raise RuntimeError('Run configuration mismatch. Use a distinct --out directory.')
    setting_path.write_text(json.dumps(settings, indent=2))
    done = set()
    if pred_path.exists():
        done = {json.loads(s)['sample_id'] for s in pred_path.open() if s.strip()}
    todo = [i for i in ids if i not in done]
    print(f'{run}: {len(ids)} target samples, {len(todo)} remaining')
    if todo:
        ds = load_dataset(repo, str(args.context_length), split='test', revision=revision)
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        with adapter, pred_path.open('a') as f:
            for n, i in enumerate(todo, 1):
                row = ds[i]
                answers = ast.literal_eval(row['answer']) if isinstance(row['answer'], str) else list(row['answer'])
                rec = infer(pipe, adapter, row['context'], row['question'], row['answer_prefix'],
                            int(row['max_new_tokens']), args.chunk_size)
                rec.update({'sample_id': i, 'task': row['task'],
                            'score': score(rec['pred'], answers, row['task']), 'answer': answers})
                f.write(json.dumps(rec) + '\n'); f.flush()
                if n % 10 == 0 or n == len(todo):
                    print(f'  {n}/{len(todo)} latest {rec["latency_s"]:.1f}s '
                          f'physical_KV={rec["kv_cache_bytes"]/1e9:.3f}GB')
        with (out_dir / 'invocations.jsonl').open('a') as f:
            f.write(json.dumps({'started':started.isoformat(),
                                'wall_s':time.perf_counter()-t0,
                                'samples':len(todo),'per_task':args.per_task})+'\n')
    invocations_file = out_dir / 'invocations.jsonl'
    invocations = [json.loads(x) for x in invocations_file.read_text().splitlines()] if invocations_file.exists() else []
    summary_json(out_dir, run, args, pred_path, invocations, repr(adapter), pipe.model)


if __name__ == '__main__':
    main()
