# base-bf16 on dev (99 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.494 | [0.455, 0.534] |
| Field F1, numeric | 0.491 | [0.459, 0.524] |
| Field F1, text | 0.505 | [0.417, 0.590] |
| Field F1, numeric (lenient) | 0.530 | [0.499, 0.561] |
| Precision | 0.368 | [0.335, 0.404] |
| Recall | 0.748 | [0.702, 0.794] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.000 | [0.000, 0.000] |
| TED accuracy | 0.376 | [0.320, 0.427] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 337 / 708 / 1335 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 15.31 / 30.65 |
| Output tokens per second of wall time | 22.5 |
| Peak VRAM (MB) | 7378 |
