# CORD v2 data report

| Split | Receipts | Fields | Numeric share | Fields/receipt p50 (max) |
|---|---|---|---|---|
| train | 773 | 10613 | 78% | 11 (71) |
| dev | 99 | 1187 | 78% | 11 (40) |
| test | 100 | 1301 | 78% | 11 (43) |

| Split | Target tokens p50 | p95 | max |
|---|---|---|---|
| train | 99 | 262 | 573 |
| dev | 100 | 177 | 323 |
| test | 102 | 233 | 368 |

## Leakage check
- Identical images across splits: **0**
- Cross-split pairs within 20 of 256 hash bits: **28**
- Distance from each dev/test receipt to its nearest receipt in another split: {'min': 0.0, 'p50': 78.0, 'p95': 97.0, 'max': 102.0}
- Identical images within a split: 2
- Receipts before exclusion: {'train': 800, 'dev': 100, 'test': 100}
- **Excluded as copies of a receipt in a later split: 28** (train copy dropped; dev copy dropped for a dev/test pair; test kept whole). The split table above counts the kept receipts.

| Excluded | Copy of | Hash distance |
|---|---|---|
| cord-train-0006 | cord-test-0095 | 4 |
| cord-train-0015 | cord-validation-0035 | 0 |
| cord-train-0044 | cord-test-0075 | 0 |
| cord-train-0056 | cord-validation-0065 | 1 |
| cord-train-0084 | cord-validation-0053 | 2 |
| cord-train-0107 | cord-test-0086 | 0 |
| cord-train-0133 | cord-test-0025 | 7 |
| cord-train-0221 | cord-validation-0006 | 0 |
| cord-train-0234 | cord-validation-0030 | 0 |
| cord-train-0251 | cord-test-0050 | 1 |
| cord-train-0283 | cord-validation-0028 | 3 |
| cord-train-0325 | cord-validation-0045 | 1 |
| cord-train-0353 | cord-validation-0036 | 2 |
| cord-train-0354 | cord-test-0079 | 0 |
| cord-train-0420 | cord-validation-0098 | 0 |
| cord-train-0426 | cord-test-0029 | 1 |
| cord-train-0428 | cord-test-0044 | 0 |
| cord-train-0483 | cord-validation-0097 | 5 |
| cord-train-0489 | cord-validation-0067 | 0 |
| cord-train-0589 | cord-test-0028 | 1 |
| cord-train-0627 | cord-validation-0096 | 2 |
| cord-train-0648 | cord-test-0067 | 3 |
| cord-train-0733 | cord-validation-0089 | 0 |
| cord-train-0765 | cord-test-0080 | 1 |
| cord-train-0766 | cord-test-0069 | 2 |
| cord-train-0785 | cord-test-0031 | 0 |
| cord-train-0797 | cord-test-0033 | 1 |
| cord-validation-0059 | cord-test-0038 | 0 |
