# bf16-merged-vllm on dev (99 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.910 | [0.877, 0.938] |
| Field F1, numeric | 0.924 | [0.894, 0.950] |
| Field F1, text | 0.861 | [0.790, 0.918] |
| Field F1, numeric (lenient) | 0.973 | [0.962, 0.984] |
| Precision | 0.910 | [0.878, 0.937] |
| Recall | 0.910 | [0.877, 0.939] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.455 | [0.354, 0.556] |
| TED accuracy | 0.958 | [0.941, 0.973] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 102 / 177 / 318 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 2.04 / 3.34 |
| Output tokens per second of wall time | 50.4 |
| Peak VRAM (MB) | — |
