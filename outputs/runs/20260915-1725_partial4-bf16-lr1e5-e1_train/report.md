# partial4-bf16-lr1e5-e1 on dev (99 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.753 | [0.701, 0.802] |
| Field F1, numeric | 0.776 | [0.731, 0.819] |
| Field F1, text | 0.668 | [0.575, 0.760] |
| Field F1, numeric (lenient) | 0.818 | [0.775, 0.860] |
| Precision | 0.772 | [0.715, 0.830] |
| Recall | 0.735 | [0.681, 0.787] |
| Valid JSON | 0.990 | [0.970, 1.000] |
| Receipt exact match | 0.162 | [0.091, 0.232] |
| TED accuracy | 0.804 | [0.760, 0.846] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 1.0% |
| Output tokens p50 / p95 / max | 94 / 200 / 1024 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 4.69 / 9.67 |
| Output tokens per second of wall time | 20.3 |
| Peak VRAM (MB) | 8638 |
