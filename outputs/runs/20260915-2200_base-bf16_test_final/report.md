# base-bf16 on test (100 receipts)

| Metric | Value | 95% interval |
|---|---|---|
| Field F1 | 0.440 | [0.395, 0.484] |
| Field F1, numeric | 0.444 | [0.406, 0.484] |
| Field F1, text | 0.422 | [0.340, 0.504] |
| Field F1, numeric (lenient) | 0.487 | [0.446, 0.528] |
| Precision | 0.326 | [0.289, 0.365] |
| Recall | 0.674 | [0.616, 0.727] |
| Valid JSON | 0.990 | [0.970, 1.000] |
| Receipt exact match | 0.000 | [0.000, 0.000] |
| TED accuracy | 0.362 | [0.309, 0.412] |

| Run | Value |
|---|---|
| Truncated at max_new_tokens | 1.0% |
| Output tokens p50 / p95 / max | 340 / 992 / 2048 |
| Prompt tokens p50 | 1270 |
| Latency p50 / p95 (s) | 14.33 / 37.80 |
| Output tokens per second of wall time | 24.2 |
| Peak VRAM (MB) | 7378 |
