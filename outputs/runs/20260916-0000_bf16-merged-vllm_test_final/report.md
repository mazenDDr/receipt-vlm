# bf16-merged-vllm on test (100 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.852 | [0.810, 0.891] |
| Field F1, numeric | 0.870 | [0.827, 0.910] |
| Field F1, text | 0.790 | [0.721, 0.854] |
| Field F1, numeric (lenient) | 0.914 | [0.882, 0.946] |
| Precision | 0.852 | [0.810, 0.892] |
| Recall | 0.852 | [0.810, 0.891] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.400 | [0.300, 0.490] |
| TED accuracy | 0.945 | [0.928, 0.959] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 102 / 242 / 403 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 2.07 / 4.32 |
| Output tokens per second of wall time | 51.1 |
| Peak VRAM (MB) | — |
