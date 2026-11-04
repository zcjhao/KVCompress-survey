"""Dependency-light tests; run under the team's already installed torch."""
import unittest
import torch
from native_methods.duoattention.dual_cache import LayerState, DualKVCache, append_layer
from native_methods.duoattention.attention import _group_attention


def tensor(n, h=1):
    return torch.arange(n, dtype=torch.float32).view(1, 1, n, 1).expand(1, h, n, 1).contiguous()


class DualCacheTests(unittest.TestCase):
    def test_sink_and_recent_are_physically_shorter(self):
        state = LayerState()
        for start, n in [(0,3),(3,4),(7,13)]:
            kr = torch.ones(1, 1, n, 4)
            ks = torch.ones(1, 1, n, 4)
            append_layer(state, kr, kr, ks, ks, sink_size=2, recent_size=3)
        self.assertEqual(state.seen, 20)
        self.assertEqual(state.retrieval_k.shape[-2], 20)
        self.assertEqual(state.sink_k.shape[-2], 2)
        self.assertEqual(state.recent_k.shape[-2], 3)
        # full 2 heads x 20 tokens x K/V x 4 floats => 1280 bytes
        self.assertEqual(state.bytes(), (20 + 2 + 3) * 2 * 4 * 4)
        self.assertLess(state.bytes(), 2 * 20 * 2 * 4 * 4)

    def test_no_retrieval_heads(self):
        state = LayerState()
        k = torch.zeros(1, 0, 10, 8)
        s = torch.ones(1, 2, 10, 8)
        append_layer(state, k, k, s, s, 2, 3)
        self.assertEqual(state.seen, 10)
        self.assertIsNone(state.retrieval_k)
        self.assertEqual(state.sink_k.shape[2] + state.recent_k.shape[2], 5)

    def test_streaming_causal_mask_excludes_middle(self):
        # All q/k are zeros => uniform attention, output is mean of selected V.
        q = torch.zeros(1,1,1,1)
        k = torch.zeros(1,1,6,1)
        v = tensor(6)
        result = _group_attention(q,k,v,torch.tensor([5]),torch.arange(6),sink=1,recent=2)
        self.assertAlmostEqual(result.item(), (0 + 4 + 5)/3, places=5)
        full = _group_attention(q,k,v,torch.tensor([5]),torch.arange(6),sink=None,recent=None)
        self.assertAlmostEqual(full.item(), 2.5, places=5)


if __name__ == '__main__':
    unittest.main()
