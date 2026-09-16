# ft-r16-gguf-q2_k-imat on test (100 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.791 | [0.737, 0.845] |
| Field F1, numeric | 0.811 | [0.756, 0.864] |
| Field F1, text | 0.721 | [0.641, 0.804] |
| Field F1, numeric (lenient) | 0.850 | [0.801, 0.895] |
| Precision | 0.814 | [0.766, 0.861] |
| Recall | 0.770 | [0.693, 0.845] |
| Valid JSON | 0.970 | [0.930, 1.000] |
| Receipt exact match | 0.340 | [0.250, 0.430] |
| TED accuracy | 0.875 | [0.834, 0.911] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 104 / 243 / 386 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 1.51 / 2.62 |
| Output tokens per second of wall time | 76.8 |
| Peak VRAM (MB) | — |
