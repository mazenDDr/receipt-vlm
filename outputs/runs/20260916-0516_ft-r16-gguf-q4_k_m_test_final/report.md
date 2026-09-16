# ft-r16-gguf-q4_k_m on test (100 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.832 | [0.785, 0.876] |
| Field F1, numeric | 0.854 | [0.806, 0.899] |
| Field F1, text | 0.756 | [0.684, 0.828] |
| Field F1, numeric (lenient) | 0.892 | [0.852, 0.929] |
| Precision | 0.832 | [0.786, 0.877] |
| Recall | 0.832 | [0.785, 0.876] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.390 | [0.290, 0.490] |
| TED accuracy | 0.929 | [0.906, 0.950] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 102 / 242 / 384 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 1.51 / 2.47 |
| Output tokens per second of wall time | 75.4 |
| Peak VRAM (MB) | — |
