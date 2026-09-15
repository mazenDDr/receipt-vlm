# lora-bf16-r16-lm-lr4e4-e1 on dev (99 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.869 | [0.815, 0.913] |
| Field F1, numeric | 0.892 | [0.846, 0.932] |
| Field F1, text | 0.785 | [0.696, 0.866] |
| Field F1, numeric (lenient) | 0.941 | [0.897, 0.971] |
| Precision | 0.860 | [0.794, 0.911] |
| Recall | 0.878 | [0.835, 0.916] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.404 | [0.303, 0.505] |
| TED accuracy | 0.932 | [0.908, 0.953] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 102 / 180 / 413 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 8.47 / 15.49 |
| Output tokens per second of wall time | 11.5 |
| Peak VRAM (MB) | 7782 |
