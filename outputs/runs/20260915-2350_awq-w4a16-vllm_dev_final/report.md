# awq-w4a16-vllm on dev (99 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.874 | [0.830, 0.913] |
| Field F1, numeric | 0.896 | [0.859, 0.931] |
| Field F1, text | 0.798 | [0.708, 0.880] |
| Field F1, numeric (lenient) | 0.954 | [0.926, 0.975] |
| Precision | 0.878 | [0.835, 0.916] |
| Recall | 0.870 | [0.824, 0.910] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.455 | [0.364, 0.556] |
| TED accuracy | 0.944 | [0.925, 0.962] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 99 / 171 / 336 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 1.01 / 1.53 |
| Output tokens per second of wall time | 100.0 |
| Peak VRAM (MB) | — |
