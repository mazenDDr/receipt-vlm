# qlora-r16-lm-lr4e4 on test (100 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.836 | [0.792, 0.878] |
| Field F1, numeric | 0.863 | [0.820, 0.905] |
| Field F1, text | 0.740 | [0.670, 0.812] |
| Field F1, numeric (lenient) | 0.896 | [0.859, 0.933] |
| Precision | 0.839 | [0.796, 0.881] |
| Recall | 0.832 | [0.786, 0.875] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.390 | [0.290, 0.490] |
| TED accuracy | 0.935 | [0.915, 0.953] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 102 / 237 / 424 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 13.22 / 29.91 |
| Output tokens per second of wall time | 7.8 |
| Peak VRAM (MB) | 3631 |
