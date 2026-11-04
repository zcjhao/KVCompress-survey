"""Small random Llama integration test (no model download; run in team venv)."""
import unittest
import torch

try:
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:
    LlamaConfig = LlamaForCausalLM = None

from native_methods.duoattention.attention import NativeDuoAttention


@unittest.skipIf(LlamaConfig is None, 'Transformers not installed (run in team venv)')
class SmallLlamaTest(unittest.TestCase):
    def test_full_group_matches_hf_and_native_eviction(self):
        torch.manual_seed(7)
        cfg = LlamaConfig(vocab_size=128,hidden_size=64,intermediate_size=128,
                          num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                          max_position_embeddings=256,attention_dropout=0.0)
        model = LlamaForCausalLM(cfg).eval().to('cpu')
        ids = torch.randint(0, 127, (1, 19))
        with torch.inference_mode():
            reference = model(input_ids=ids, logits_to_keep=1, use_cache=False).logits
            all_retrieval = torch.ones((2,2), dtype=torch.bool)
            with NativeDuoAttention(model, all_retrieval, sink_size=2, recent_size=3) as native:
                for start in range(0, 19, 5):
                    out = model(input_ids=ids[:, start:start+5], use_cache=False,
                                cache_position=torch.arange(start,min(start+5,19)), logits_to_keep=1)
                self.assertLess((reference - out.logits).abs().max().item(), 5e-4)
                self.assertEqual(native.cache.breakdown()['seen_tokens'], 19)
            selected = torch.tensor([[True, False], [False, True]])
            with NativeDuoAttention(model, selected, sink_size=2, recent_size=3) as native:
                for start in range(0, 19, 5):
                    out = model(input_ids=ids[:, start:start+5], use_cache=False,
                                cache_position=torch.arange(start,min(start+5,19)), logits_to_keep=1)
                measured = native.cache.bytes()
                full = 2 * 2 * 19 * (cfg.hidden_size // cfg.num_attention_heads) * 2 * 4
                self.assertLess(measured, full)
                self.assertEqual(native.cache.layers[0].sink_k.shape[-2], 2)
                self.assertEqual(native.cache.layers[0].recent_k.shape[-2], 3)
                self.assertTrue(torch.isfinite(out.logits).all().item())
            # original forward functions were restored
            restored = model(input_ids=ids, logits_to_keep=1, use_cache=False).logits
            self.assertLess((reference - restored).abs().max().item(), 5e-4)


if __name__ == '__main__':
    unittest.main()
