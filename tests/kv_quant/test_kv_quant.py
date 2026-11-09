"""CPU tests for the quantized KV caches (scripts/kv_quant.py). No GPU, model download or kvpress needed:

    python -m unittest discover -s tests/kv_quant -v
"""
import sys
import unittest
from pathlib import Path

import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from common import cache_bytes  # noqa: E402
from kv_quant import FP8Layer, KIVILayer, QuantizedKVCache, dequantize, pack, quantize, unpack  # noqa: E402


def tiny_llama():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
    return LlamaForCausalLM(config).to(torch.bfloat16).eval()


@torch.inference_mode()
def pipeline_run(model, cache, context, question, new_tokens=8):
    """The kvpress pipeline's sequence: prefill the context through model.model, then the question with
    explicit positions, then greedy decoding one token at a time. Returns all logits and the ids."""
    model.model(input_ids=context, past_key_values=cache)
    pos = torch.arange(context.shape[1], context.shape[1] + question.shape[1]).unsqueeze(0)
    out = model(input_ids=question, past_key_values=cache, position_ids=pos, logits_to_keep=1)
    logits, ids = [out.logits[0, -1]], [out.logits[0, -1].argmax()]
    pos = pos[:, -1:] + 1
    for i in range(new_tokens - 1):
        out = model(input_ids=ids[-1].view(1, 1), past_key_values=cache, position_ids=pos + i)
        logits.append(out.logits[0, -1])
        ids.append(out.logits[0, -1].argmax())
    return torch.stack(logits), torch.stack(ids)


class TestPacking(unittest.TestCase):
    def test_roundtrip(self):
        for nbits in (2, 4):
            q = torch.randint(0, 1 << nbits, (2, 3, 5, 64), dtype=torch.uint8)
            p = pack(q, nbits)
            self.assertEqual(p.shape[-1], 64 * nbits // 8)
            self.assertTrue(torch.equal(unpack(p, nbits), q))

    def test_quantization_error_within_half_step(self):
        x = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16) * 3
        for nbits in (2, 4):
            for per_channel in (True, False):
                packed, scale, mn = quantize(x, nbits, 32, per_channel)
                y = dequantize(packed, scale, mn, nbits, 32, per_channel, torch.float32)
                # Half a quantization step, plus slack for FP16 scales and BF16 inputs.
                step = scale.float().max().item()
                self.assertLessEqual((y - x.float()).abs().max().item(), 0.5 * step + 0.05)

    def test_key_scales_are_per_channel_value_scales_per_token(self):
        x = torch.randn(1, 2, 64, 64)
        _, k_scale, _ = quantize(x, 4, 32, per_channel=True)
        _, v_scale, _ = quantize(x, 4, 32, per_channel=False)
        self.assertEqual(tuple(k_scale.shape), (1, 2, 2, 1, 64))  # 2 token groups x 64 channels
        self.assertEqual(tuple(v_scale.shape), (1, 2, 64, 2, 1))  # 64 tokens x 2 channel groups


class TestKIVILayer(unittest.TestCase):
    def test_residual_and_lengths(self):
        layer = KIVILayer(4, group_size=32, residual_length=128)
        k, v = torch.randn(1, 2, 300, 64), torch.randn(1, 2, 300, 64)
        out_k, _ = layer.update(k, v)
        self.assertTrue(torch.equal(out_k, k))  # prefill attends to exact keys
        self.assertEqual((layer.n_quantized, layer.keys.shape[-2]), (256, 44))
        for i in range(100):
            out_k, out_v = layer.update(torch.randn(1, 2, 1, 64), torch.randn(1, 2, 1, 64))
            self.assertEqual(out_k.shape[-2], 301 + i)
            self.assertEqual(layer.get_seq_length(), 301 + i)
        self.assertEqual((layer.n_quantized, layer.keys.shape[-2]), (384, 16))  # flushed once at 128

    def test_stored_bytes(self):
        t = 1024
        bf16 = 2 * 2 * t * 64 * 2  # K and V, 2 heads, 64 dims, 2 bytes
        for nbits, expected in ((4, 0.5 + 0.125), (2, 0.25 + 0.125)):
            layer = KIVILayer(nbits)
            layer.update(torch.randn(1, 2, t, 64, dtype=torch.bfloat16), torch.randn(1, 2, t, 64, dtype=torch.bfloat16))
            # Per element: nbits/8 data + per-group FP16 scale and zero point (4 bytes / 32 elements), averaged over K and V.
            self.assertAlmostEqual(layer.stored_bytes() / bf16, expected / 2, places=6)


class TestWithModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_llama()
        g = torch.Generator().manual_seed(1)
        cls.context = torch.randint(0, 128, (1, 600), generator=g)
        cls.question = torch.randint(0, 128, (1, 12), generator=g)
        cls.ref_logits, cls.ref_ids = pipeline_run(cls.model, DynamicCache(), cls.context, cls.question)

    def run_cache(self, make_layer):
        cache = QuantizedKVCache(self.model.config, make_layer)
        logits, ids = pipeline_run(self.model, cache, self.context, self.question)
        return cache, logits, ids

    def test_short_context_is_exact(self):
        """Fewer tokens than the residual window: nothing is quantized, so KIVI equals the full cache."""
        ctx = self.context[:, :100]
        ref, _ = pipeline_run(self.model, DynamicCache(), ctx, self.question[:, :10])
        cache = QuantizedKVCache(self.model.config, lambda: KIVILayer(2))
        out, _ = pipeline_run(self.model, cache, ctx, self.question[:, :10])
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_close_to_full_cache(self):
        """Lossy, and more bits means less error: FP8 < int4 < int2, all well below the logits' spread."""
        errs = []
        for make_layer in (FP8Layer, lambda: KIVILayer(4), lambda: KIVILayer(2)):
            cache, logits, _ = self.run_cache(make_layer)
            self.assertEqual(cache.get_seq_length(), 600 + 12 + 7)
            errs.append((logits - self.ref_logits).float().abs().mean().item())
        spread = self.ref_logits.float().std().item()
        self.assertTrue(0 < errs[0] < errs[1] < errs[2] < 0.5 * spread, f"mean logit errors {errs}, spread {spread}")

    def test_cache_bytes(self):
        full = DynamicCache()
        with torch.inference_mode():
            self.model.model(input_ids=self.context, past_key_values=full)
        # 600 tokens: 512 quantized (4 x 128) and 88 still in the BF16 residual window.
        for make_layer, per_quantized in ((FP8Layer, None), (lambda: KIVILayer(4), 0.3125), (lambda: KIVILayer(2), 0.1875)):
            cache = QuantizedKVCache(self.model.config, make_layer)
            with torch.inference_mode():
                self.model.model(input_ids=self.context, past_key_values=cache)
            expected = 0.5 if per_quantized is None else (512 * per_quantized + 88) / 600
            self.assertAlmostEqual(cache_bytes(cache) / cache_bytes(full), expected, places=6, msg=repr(cache))

if __name__ == "__main__":
    unittest.main()
