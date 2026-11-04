import unittest
import torch
import torch.nn.functional as F
from native_methods.duoattention.attention import _group_attention

class RetrievalFlashBiasTests(unittest.TestCase):
    def test_retrieval_matches_explicit_causal_mask_cpu(self):
        # Includes non-square chunk prefill and single-token decode.
        for previous, chunk in ((0, 8), (9, 4), (17, 1)):
            with self.subTest(previous=previous, chunk=chunk):
                torch.manual_seed(21)
                q = torch.randn(1, 4, chunk, 16)
                k = torch.randn(1, 2, previous + chunk, 16)
                v = torch.randn_like(k)
                q_pos = torch.arange(previous, previous + chunk)
                k_pos = torch.arange(previous + chunk)
                actual = _group_attention(q, k, v, q_pos, k_pos,
                                          sink=None, recent=None)
                ek = k.repeat_interleave(2, dim=1)
                ev = v.repeat_interleave(2, dim=1)
                ref_mask = (k_pos[None,:] <= q_pos[:,None])[None,None,:,:]
                expected = F.scaled_dot_product_attention(q, ek, ev,
                                                           attn_mask=ref_mask,
                                                           dropout_p=0.0)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_streaming_keeps_original_mask(self):
        q = torch.randn(1, 4, 3, 16)
        k = torch.randn(1, 2, 7, 16)
        v = torch.randn_like(k)
        q_pos = torch.arange(8, 11)
        k_pos = torch.tensor([0, 1, 5, 6, 8, 9, 10])
        actual = _group_attention(q,k,v,q_pos,k_pos,sink=2,recent=3)
        valid = (k_pos[None,:] <= q_pos[:,None]) & ((k_pos[None,:] < 2) | (k_pos[None,:] >= q_pos[:,None]-2))
        ref = F.scaled_dot_product_attention(q, k.repeat_interleave(2,dim=1), v.repeat_interleave(2,dim=1), attn_mask=valid[None,None,:,:], dropout_p=0.)
        torch.testing.assert_close(actual, ref, atol=1e-5, rtol=1e-5)
