# ft-r16-gguf-bf16 on test (100 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.846 | [0.801, 0.891] |
| Field F1, numeric | 0.866 | [0.820, 0.910] |
| Field F1, text | 0.774 | [0.703, 0.846] |
| Field F1, numeric (lenient) | 0.910 | [0.871, 0.946] |
| Precision | 0.848 | [0.803, 0.892] |
| Recall | 0.844 | [0.799, 0.889] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.410 | [0.310, 0.500] |
| TED accuracy | 0.940 | [0.921, 0.957] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 102 / 242 / 403 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 2.57 / 4.87 |
| Output tokens per second of wall time | 41.4 |
| Peak VRAM (MB) | — |
