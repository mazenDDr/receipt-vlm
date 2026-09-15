# partial4-bf16-lr5e5-e1 on dev (99 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.817 | [0.762, 0.866] |
| Field F1, numeric | 0.845 | [0.798, 0.887] |
| Field F1, text | 0.718 | [0.619, 0.818] |
| Field F1, numeric (lenient) | 0.888 | [0.845, 0.925] |
| Precision | 0.823 | [0.766, 0.874] |
| Recall | 0.812 | [0.756, 0.860] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.323 | [0.232, 0.414] |
| TED accuracy | 0.895 | [0.865, 0.921] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 101 / 181 / 310 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 4.44 / 7.79 |
| Output tokens per second of wall time | 22.2 |
| Peak VRAM (MB) | 8638 |
