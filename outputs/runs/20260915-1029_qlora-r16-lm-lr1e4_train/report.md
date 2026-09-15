# qlora-r16-lm-lr1e4 on dev (99 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.869 | [0.823, 0.907] |
| Field F1, numeric | 0.893 | [0.856, 0.925] |
| Field F1, text | 0.787 | [0.692, 0.871] |
| Field F1, numeric (lenient) | 0.936 | [0.906, 0.961] |
| Precision | 0.870 | [0.824, 0.909] |
| Recall | 0.869 | [0.822, 0.907] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.394 | [0.303, 0.485] |
| TED accuracy | 0.931 | [0.910, 0.950] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 104 / 177 / 302 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 12.95 / 22.23 |
| Output tokens per second of wall time | 8.2 |
| Peak VRAM (MB) | 3867 |
