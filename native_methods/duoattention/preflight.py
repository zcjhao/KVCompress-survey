"""Environment + official-pattern preflight. No model download or GPU computation."""
from __future__ import annotations
import importlib.metadata as md
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from presses.duoattention_utils import read_pattern, method_config, budget_plan, binary_heads


def main():
    for name in ('torch','transformers','kvpress','datasets','pandas'):
        print(name, md.version(name))
    import torch, transformers
    assert transformers.__version__ == '5.2.0', 'Pin mismatch; DO NOT modify team lock file'
    print('CUDA runtime:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
    scores, sink, recent = read_pattern()
    plan = budget_plan(32768, 0.5, sink, recent)
    mask = binary_heads(scores, plan['streaming_heads'])
    print('Official head-pattern hash verified:', method_config()['pattern_sha256'])
    print('Retrieval head groups:', int(mask.sum()), 'Streaming groups:', plan['streaming_heads'])
    print('Estimated retained KV fraction:', round(plan['effective_retained_ratio'],5))
    print('Preflight complete: does NOT establish correctness on the A100')

if __name__ == '__main__':
    main()
