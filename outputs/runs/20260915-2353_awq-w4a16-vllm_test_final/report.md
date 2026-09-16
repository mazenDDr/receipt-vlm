# awq-w4a16-vllm on test (100 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.847 | [0.805, 0.887] |
| Field F1, numeric | 0.866 | [0.824, 0.907] |
| Field F1, text | 0.780 | [0.710, 0.847] |
| Field F1, numeric (lenient) | 0.908 | [0.873, 0.940] |
| Precision | 0.853 | [0.810, 0.893] |
| Recall | 0.842 | [0.799, 0.882] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.400 | [0.310, 0.500] |
| TED accuracy | 0.941 | [0.924, 0.958] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 102 / 235 / 349 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 1.05 / 1.98 |
| Output tokens per second of wall time | 103.1 |
| Peak VRAM (MB) | — |
