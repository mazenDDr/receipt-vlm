# How CORD labels receipts, and where the model disagrees

Thirty dev receipts were read against their photographs by the repository owner, and every pattern found
that way was then counted across all 99 dev receipts (`scripts/audit_patterns.py`). This document is the
result: the labelling conventions worth knowing, how often the model departs from each, and the labels
that look wrong.

The model audited is the deployed one — QLoRA r16, LM-only, lr 4e-4, dev field F1 0.900.

## Why this matters for the numbers

Scoring is strict by design: values are compared as strings, because digits are the point. That means a
label convention the model does not follow costs exactly as much as a misread digit. Of 73 disagreeing
key paths in the audited thirty, **26 (36%) were separator or punctuation conventions**, not misreadings.
The strict/lenient gap on the test split says the same thing at scale: numeric F1 **0.866 strict** against
**0.910 lenient**.

## The conventions

### 1. Units stay with the count, not the name

The paper prints `1Prs Sop Sui Jiao`. CORD labels it `cnt: "1Prs"`, `nm: "Sop Sui Jiao"` — the unit belongs
to the quantity. The model instead splits on character class: digits into `cnt`, letters onto the front of
the name, giving `cnt: "1"` and `nm: "Prs Sop Sui Jiao"`. Both fields are then wrong, so one habit costs
two fields per item.

Measured: 1 receipt, 7 fields — concentrated rather than common. All of `cord-validation-0025`.

### 2. A count may be printed anywhere on the line

`Cha Keaw L... x2` puts the quantity after the name. CORD labels `cnt: "x2"`. The model looks to the right
of the line for a number, finds none, and emits no count at all.

Measured: 1 receipt, 2 fields (`cord-validation-0055`).

### 3. Menu items are flat unless the receipt nests them

CORD keeps modifiers and follow-on lines inside the item they belong to. The model adds a `sub` object the
label does not have, or flattens one the label does. Either way every child field lands under a different
key path, and one bracket can cost a whole item.

Measured: 5 receipts where the model added nesting, 2 where it removed it.

### 4. The label transcribes what is printed, including stray marks

Gold keeps `':9,000'` with the colon, `'@ 12.000'` with the space, `'1 x'` with the gap. The model tidies
them: `'9,000'`, `'@12.000'`, `'1x'`. The label is a transcription, not a normalization, and the model is
being penalised for cleaning up.

Measured: **8 receipts, 18 fields — the largest single category.**

### 5. Thousands separators follow the glyph on the paper, and are not consistent

Indonesian receipts use `.` and `,` interchangeably. The label follows whatever is printed, which means the
convention changes between receipts and sometimes within one.

Measured, and this is where the hand audit and the counts disagree: the flip runs **both ways** —
3 receipts read `,` as `.` (12 fields), and 5 receipts read `.` as `,` (6 fields). A model with a
systematic glyph bias would flip one direction. Flipping both directions means some of these are the
model misreading and some are the labels disagreeing with each other, and the two cannot be separated
without the photographs.

Known label inconsistency: `cord-validation-0017` writes `20,000` for every price and `80.000` for
`changeprice` on the same receipt.

### 6. Names keep the spacing that is printed

`LEMONADE22OZ` stays closed up. The model inserts a space. Same characters, different string, scored wrong.

Measured: 4 receipts, 5 fields — in 3 the model added spaces, in 1 the label has more.

### 7. Payment sections are distinguished by which line they sit on

`sub_total.discount_price`, `sub_total.tax_price`, `total.cashprice`, `total.creditcardprice`,
`total.emoneyprice` and `total.total_etc` are separate keys, and the model confuses the discount section
with the tax and payment section. The value read is correct; the section it is filed under is not.

Measured: 3 receipts — `total.total_etc` → `total.emoneyprice`, `total.cashprice` → `total.total_price`,
`total.total_etc` → `total.creditcardprice`.

## Labels that look wrong

These need a photograph to settle, and the ones below were flagged during the audit. They are **not**
counted as model errors anywhere in this repository's results.

| Receipt | Label | Model | Why it is doubted |
|---|---|---|---|
| `cord-validation-0017` | `20,000` throughout, `80.000` for change | `20.000` throughout | Two separator conventions on one receipt |
| `cord-test-0016` | `IVINERAL WATER (bundling) 5k` | `MINERAL WATER (bundling) 5k` | Reads as a literal transcription of an `M` printed as `IVI` |
| `cord-test-0061` | `-2 600`, `-3,200`, `-7.200` | — | Three separator conventions in one receipt |
| `cord-test-0012` | `menuqty_cnt: "(2"` | `menuqty_cnt: "2"` | Unclosed parenthesis in the label |

## What follows from this

**The remaining headroom is conventions and structure, not reading.** Adding training data will not teach
the model that `@ 12.000` keeps its space or that `1Prs` is a count. Three things would:

1. **Show the convention in the prompt.** The instruction names every CORD key but says nothing about
   transcribing stray marks, keeping units with counts, or leaving spacing alone.
2. **Constrain the structure at decode time.** Nesting and section-key errors are schema decisions, and a
   schema-aware decoder can rule out the invalid ones rather than hoping the model learns them.
3. **Report the lenient numeric metric beside the strict one**, always. The strict number is the honest
   headline; the gap between them is how much of the remaining error is punctuation.

Reproduce the counts with:

```bash
python scripts/audit_patterns.py --run outputs/runs/<dev run> --out outputs/label_audit/patterns.json
```
