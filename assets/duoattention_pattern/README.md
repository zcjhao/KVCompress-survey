# Frozen DuoAttention head pattern

The DuoAttention experiment loads `config.json` and `full_attention_heads.tsv`
directly from this directory. These are active runtime assets, unchanged from:

- Repository: https://github.com/mit-han-lab/duo-attention
- Commit: `fe93c314ae87306ef6629dc16713250b4718ffe7`
- Original directory: `attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10`
- TSV SHA256: `042e0c23e89278525928bf45b6eba3200d9bf4a65231cd469d69eb9b98c61218`
- Shape: 32 layers x 8 KV heads; sink 128, recent 256.

The upstream MIT license is included in `LICENSE`. The training and native
dual-cache source tree is not required by this project's kvpress adapter.
