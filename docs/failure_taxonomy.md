# What the fine-tuned model gets wrong

The 20 worst receipts of the test split, read by hand against their gold labels, plus mechanical counts
over all 100. The model scored is the deployed one: QLoRA r16, LM-only, lr 4e-4, scored in transformers
(`outputs/runs/20260915-2231_qlora-r16-lm-lr4e4_test_final`, test field F1 **0.836**).

The headline: **structure and formatting dominate, perception is comparatively rare.** Every one of the
100 test receipts produced valid JSON, so there are no format failures at all in the usual sense.

## Counts over all 100 test receipts

Detected mechanically (`nesting` = gold and prediction disagree on whether any `sub` nesting exists;
`separator/whitespace only` = every value matches once `.`, `,` and spaces are stripped):

| Category | Receipts | Share |
|---|---|---|
| Exactly correct | 41 | 41% |
| Digit difference somewhere | 16 | 16% |
| Separator or whitespace only | 15 | 15% |
| Nesting error | 9 | 9% |
| Item code folded into the name | 2 | 2% |
| Invalid JSON / truncated | **0** | **0%** |

Categories overlap: one receipt can be both a nesting error and a digit difference.

**Formatting is worth about a third of the numeric gap.** Strict numeric F1 on test is 0.866 and the
lenient variant — which keeps digits but ignores separators, currency and trailing zero cents — is 0.910.

## The categories, with examples

### 1. Nesting: a flat list read as nested, or the reverse (9%)

The most expensive category, because one misplaced bracket moves every child field to a wrong key path
and each one is scored as both a false positive and a false negative.

- **`cord-test-0059`** (F1 0.35, the second-worst receipt). Items 1 and 2 are correct; items 3–8 are read
  correctly and then nested as `sub` of item 3. Every name and price is right. The receipt still scores
  0.35 because the key path of six items changed.
- **`cord-test-0099`** (F1 0.41) and **`cord-test-0016`** (F1 0.52) are the mirror image: gold nests six
  and three items under a parent, the model flattens them into a list.
- **`cord-test-0088`**, **`cord-test-0031`** invent a `sub` where gold has two sibling items.

### 2. Right value, wrong key (the second-largest source of lost F1)

The value is read correctly and placed under a key CORD does not use there.

- **`cord-test-0039`** (F1 0.34, the worst receipt). `itemsubtotal` → `discountprice` on all three items,
  the item code folded into `nm` (`"HPL754BR LUNCH BOX 3P SET..."` where gold has `nm` plus `num`),
  `"PCS"` appended to counts, and a `sub_total` of `{"subtotal_price": "17", "tax_price": "17"}`
  hallucinated out of the quantity line.
- **`cord-test-0026`** (F1 0.69, text F1 **0.00**). Gold has `nm: "Nasi Cap Cay Porsi Besar"` with
  `etc: "Cap Cay"`; the model swaps them, demoting the real name into `sub`. Every number is correct.
- **`cord-test-0050`**: `changeprice` → `emoneyprice`. **`cord-test-0012`**: `itemsubtotal` → `price`.

### 3. Separator and whitespace differences only (15%)

Scored as wrong under Donut-compatible strict matching, which compares values as strings.

- **`cord-test-0024`**: `Rp.56.000` → `Rp. 56.000`, on all six values. One inserted space each.
- **`cord-test-0083`**, **`cord-test-0092`**: `20,909` → `20.909`, `10,000` → `10.000`. Indonesian
  receipts use `.` and `,` interchangeably as thousands separators, and gold follows the printed glyph.
- **`cord-test-0042`**: `15.000` → `15,000` *and* `1,500` → `1.500` on the same receipt — in opposite
  directions.
- **`cord-test-0000`**: `"-TICKET CP"` → `"- TICKET CP"`, one space.

### 4. Misread text (names)

- **`cord-test-0050`**: `"Superice Cauburry"` → `"SpecPrice Cadburry"`.
- **`cord-test-0099`**: `"French Vanilla"` → `"French Vanila"`.
- **`cord-test-0061`**: `"AREM - AREM"` → `"AREM - AREH"`.

### 5. Misread digits (16%, and rarer than expected in the worst cases)

- **`cord-test-0091`**: `cnt` 6 → 8, and signs invented: `6.000` → `-6.000`, `21.000` → `-21.000`.
- **`cord-test-0000`**: item code `901016` → `901015`.
- **`cord-test-0061`**: `unitprice` `8000` → `3000` and `9000` → `3000`.
- **`cord-test-0050`**: `tax_price` `1,773` → `1,727`.

### 6. Suspected gold-label problems — for the owner to judge against the images

These are **not** counted as model errors above, and I cannot settle them without looking at the photos.

- **`cord-test-0016`**: gold reads `"IVINERAL WATER (bundling) 5k"`; the model wrote
  `"MINERAL WATER (bundling) 5k"`. Gold looks like a literal transcription of an `M` printed as `IVI`.
  If gold is an OCR artifact, the model is right and is being penalised for it.
- **`cord-test-0061`**: gold's own `discountprice` values use three separator conventions on one
  receipt — `"-2 600"`, `"-3,200"`, `"-7.200"`. At least one is likely a labelling slip.
- **`cord-test-0012`**: gold's `menuqty_cnt` is `"(2"`, an unclosed parenthesis.

## What this means for the work

1. **Do not chase OCR.** Only a handful of the worst receipts fail because a character was misread. The
   model reads these receipts well; it misfiles what it reads.
2. **The remaining headroom is structural.** Nesting and key-choice errors are where the lost F1 is, and
   both are decisions about CORD's schema rather than about the image — which is what a schema-aware
   decoder or a constrained-decoding pass would address, not more training data.
3. **Report the lenient numeric metric next to the strict one.** 15% of receipts differ only in
   separators, an artefact of the label convention rather than a reading failure.
4. **Quantization does not change this picture** down to 4 bits: the same receipts fail the same way.
   Below 4 bits the failure mode changes entirely — see `docs/tradeoffs.md`.
