# qlora-r16-lm on dev (99 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.885 | [0.840, 0.921] |
| Field F1, numeric | 0.910 | [0.874, 0.941] |
| Field F1, text | 0.800 | [0.707, 0.881] |
| Field F1, numeric (lenient) | 0.952 | [0.924, 0.975] |
| Precision | 0.887 | [0.844, 0.922] |
| Recall | 0.883 | [0.835, 0.922] |
| Valid JSON | 1.000 | [1.000, 1.000] |
| Receipt exact match | 0.475 | [0.374, 0.576] |
| TED accuracy | 0.948 | [0.929, 0.965] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 0.0% |
| Output tokens p50 / p95 / max | 101 / 180 / 346 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 11.50 / 22.12 |
| Output tokens per second of wall time | 8.6 |
| Peak VRAM (MB) | 3628 |
