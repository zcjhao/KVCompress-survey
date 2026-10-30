# Token eviction (SnapKV): key observations

**Setup.** Llama-3.1-8B-Instruct on one A100 80 GB, kvpress 0.5.5 `SnapKVPress` (window 64, kernel 5).
RULER at 32K, 64K and 128K context (650 / 650 / 260 examples, the same splits as the full-cache baseline).
Compression ratio *r* is the fraction of the KV cache evicted, so r = 0.5 keeps half of the context tokens.

## Results

| RULER accuracy (%) | 32K | 64K | 128K | KV cache at 128K | Peak GPU memory at 128K |
|---|---|---|---|---|---|
| Full cache | 88.7 | 86.7 | 77.3 | 17.2 GB | 48.9 GB |
| SnapKV r = 0.25 | 80.2 | 72.2 | 59.5 | 12.9 GB | 44.6 GB |
| SnapKV r = 0.5 | 68.1 | 57.5 | 44.4 | 8.6 GB | 40.3 GB |
| SnapKV r = 0.75 | 46.4 | 36.2 | 30.0 | 4.3 GB | 36.0 GB |
| SnapKV r = 0.9 | 24.5 | 25.8 | 24.1 | 1.7 GB | 33.4 GB |

## Observations

1. **Accuracy falls steadily as more of the cache is evicted, with no free lunch even at mild compression.**
   Evicting only 25% already costs 8.5 points at 32K and 17.8 points at 128K.
   At r = 0.9 accuracy settles near 25% at every length; the remaining score comes almost entirely from the tasks in point 3.

2. **The loss grows with context length.** At r = 0.5 SnapKV keeps 77% of baseline accuracy at 32K, 66% at 64K and 57% at 128K.
   A fixed ratio means more absolute tokens are kept at longer lengths, but each kept token has to "win" against more competitors, and the attention scores that decide what is kept get flatter as the context grows.

3. **Which tasks break depends on what the answer looks like.**
   - *Robust:* `niah_single_1` (a needle in repetitive noise text) stays at 92 to 100% even at r = 0.75. The needle is the only unusual text, so it attracts attention and is kept.
     Aggregation tasks that depend on frequent tokens (`fwe`, `vt`) also degrade slowly, because frequent words survive eviction by being frequent.
   - *Fragile:* needles with high-entropy values and lookalike distractors collapse first.
     `niah_single_3` (a UUID value in an essay) drops from 100% to 52% at 32K and 10% at 128K at r = 0.5; `niah_multikey_2/3` (many similar keys) fall to 10 to 40%.
     A UUID spans many tokens and the model needs every one of them, so losing even part of the needle makes the answer wrong.
   - *Already weak at baseline:* `cwe` (common-word extraction) is near 0 at 64K to 128K even with the full cache, so it says little about eviction.

4. **Query-agnostic compression is a likely cause of the gap to the SnapKV paper.**
   kvpress compresses the context *before* the question is appended, so SnapKV picks tokens using the last 64 tokens of the context rather than the question.
   The paper's version selects with the question in the window. Our numbers are therefore a lower bound for SnapKV, and this should be stated when comparing with the other categories.

5. **Memory savings are exact for the KV cache and only partial for the whole GPU.**
   The KV cache shrinks exactly by (1 − r): 128 KB per token for Llama-3.1-8B in BF16, so 17.2 GB at 128K becomes 8.6 GB at r = 0.5.
   Peak memory drops by the same number of gigabytes, but the 16 GB of weights and the activations do not shrink, so peak memory falls by only 18% (48.9 to 40.3 GB) at r = 0.5.

6. **Eviction speeds up decoding only at long context, and never speeds up prefill.**
   Time to first token is unchanged (about 3.7 s at 32K, about 33 s at 128K, slightly higher than baseline because of the scoring step), since prefill still processes the full prompt.
   Time per output token at 128K drops from 47.5 ms to about 32 ms at r ≥ 0.5, but at 32K it stays at about 33 ms for every ratio: at short context decoding is limited by reading the weights, not the KV cache.

## Caveats

- One seed per setting; per-task numbers rest on 50 examples at 32K/64K and 20 at 128K, so differences of a few points on a single task are noise.
- Latency and memory were measured in the Hugging Face + kvpress harness at batch size 1, not inside a serving engine. The serving-engine throughput measurement required by the project is still to do.
- H2O, the other token eviction method, was not run in this batch.
