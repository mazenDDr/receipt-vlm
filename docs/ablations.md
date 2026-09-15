# Ablations

Dev split (99 CORD v2 receipts). Each fine-tuned arm is scored by generating the JSON and
comparing it field by field with the labels. Differences are **paired**: both arms are scored
on the same receipts, and the 95% interval comes from 2,000 bootstrap resamples of those
receipts. A difference counts as real only if its interval excludes 0. From the rank stage on,
each arm trains one epoch and is compared with a one-epoch reference, so every comparison is
like for like.

## Every run

| Variant | n | Field F1 [95% interval] | Numeric F1 | Text F1 | Precision | Recall | Valid JSON | TED | Train min | Train peak GB | Trainable M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base-bf16 | 99 | 0.494 [0.455, 0.534] | 0.491 | 0.505 | 0.368 | 0.748 | 1.000 | 0.376 | — | — | — |
| qlora-r16-lm-lr1e4 | 99 | 0.869 [0.823, 0.907] | 0.893 | 0.787 | 0.870 | 0.869 | 1.000 | 0.931 | 64 | 7.5 | 29.9 |
| qlora-r16-lm | 99 | 0.885 [0.840, 0.921] | 0.910 | 0.800 | 0.887 | 0.883 | 1.000 | 0.948 | 65 | — | 29.9 |
| qlora-r16-lm-lr4e4 | 99 | 0.900 [0.868, 0.927] | 0.922 | 0.821 | 0.902 | 0.898 | 1.000 | 0.952 | 64 | 7.5 | 29.9 |
| qlora-r16-lm-lr4e4-e1 | 99 | 0.874 [0.831, 0.911] | 0.897 | 0.794 | 0.871 | 0.877 | 1.000 | 0.935 | 26 | 7.5 | 29.9 |
| qlora-r8-lm-lr4e4-e1 | 99 | 0.843 [0.792, 0.888] | 0.869 | 0.751 | 0.848 | 0.838 | 0.990 | 0.920 | 26 | 7.2 | 15.0 |
| qlora-r32-lm-lr4e4-e1 | 99 | 0.869 [0.825, 0.908] | 0.897 | 0.773 | 0.867 | 0.872 | 1.000 | 0.947 | 26 | 7.9 | 59.9 |
| qlora-r16-lmproj-lr4e4-e1 | 99 | 0.867 [0.819, 0.908] | 0.892 | 0.780 | 0.864 | 0.870 | 1.000 | 0.936 | 26 | 7.5 | 30.2 |
| lora-bf16-r16-lm-lr4e4-e1 | 99 | 0.869 [0.815, 0.913] | 0.892 | 0.785 | 0.860 | 0.878 | 1.000 | 0.932 | 25 | 11.3 | 29.9 |
| partial4-bf16-lr1e5-e1 | 99 | 0.753 [0.701, 0.802] | 0.776 | 0.668 | 0.772 | 0.735 | 0.990 | 0.804 | 23 | 13.0 | 345.0 |
| partial4-bf16-lr5e5-e1 | 99 | 0.817 [0.762, 0.866] | 0.845 | 0.718 | 0.823 | 0.812 | 1.000 | 0.895 | 22 | 13.0 | 345.0 |

## What each choice changes

| Question | Field F1 difference [95%] | Real? | Numeric F1 | Text F1 |
|---|---|---|---|---|
| Fine-tuning at all (QLoRA r16, lr 2e-4, 2 epochs) | +0.391 [+0.337, +0.441] | yes | +0.419 [+0.376, +0.459] | +0.295 [+0.185, +0.396] |
| A. Learning rate 1e-4 instead of 2e-4 | -0.016 [-0.027, -0.004] | yes | -0.017 [-0.028, -0.007] | -0.013 [-0.040, +0.014] |
| A. Learning rate 4e-4 instead of 2e-4 | +0.015 [-0.014, +0.050] | no | +0.013 [-0.010, +0.039] | +0.021 [-0.039, +0.089] |
| One epoch instead of two (r16, lr 4e-4) | -0.026 [-0.061, +0.005] | no | -0.025 [-0.053, -0.001] | -0.027 [-0.098, +0.043] |
| B. Rank 8 instead of 16 | -0.031 [-0.069, -0.001] | yes | -0.028 [-0.064, +0.001] | -0.043 [-0.105, +0.007] |
| B. Rank 32 instead of 16 | -0.005 [-0.026, +0.009] | no | +0.000 [-0.015, +0.011] | -0.021 [-0.069, +0.011] |
| C. Also adapt the vision projector | -0.007 [-0.032, +0.009] | no | -0.005 [-0.026, +0.009] | -0.014 [-0.061, +0.015] |
| D. LoRA on a bf16 base instead of 4-bit (QLoRA) | -0.005 [-0.022, +0.009] | no | -0.004 [-0.023, +0.011] | -0.009 [-0.030, +0.015] |
| E. Partial full fine-tuning, lr 1e-5 | -0.121 [-0.167, -0.080] | yes | -0.120 [-0.160, -0.083] | -0.126 [-0.207, -0.055] |
| E. Partial full fine-tuning, lr 5e-5 | -0.057 [-0.103, -0.022] | yes | -0.051 [-0.089, -0.021] | -0.076 [-0.162, -0.013] |

Rows marked "no" are ties within noise, not evidence that the choice doesn't matter at all.
