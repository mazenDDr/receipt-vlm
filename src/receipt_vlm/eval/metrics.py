"""Field-level scoring of one prediction, and the summary of a whole run with bootstrap intervals.

Field F1 follows Donut's CORD evaluation: the receipt is flattened into a multiset of (key path, value) pairs,
a predicted pair is correct if the gold multiset still holds it, and F1 is micro-averaged over the split.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence

from receipt_vlm.eval import fields, normalize
from receipt_vlm.eval.bootstrap import interval
from receipt_vlm.eval.ted import ted_accuracy
from receipt_vlm.schemas import ExampleScore, FieldCounts, Prediction, ReceiptExample

Pairs = Counter[tuple[str, str]]
Metric = Callable[[Sequence[ExampleScore]], float]


def _pairs(data: object, norm: Callable[[str], str]) -> Pairs:
    pairs: Pairs = Counter()
    for path, value in fields.flatten(data):
        value = norm(value)
        if value:
            pairs[(path, value)] += 1
    return pairs


def _only(pairs: Pairs, kind: fields.Kind) -> Pairs:
    return Counter({pair: n for pair, n in pairs.items() if fields.kind(pair[0]) == kind})


def _counts(gold: Pairs, pred: Pairs) -> FieldCounts:
    tp = sum((gold & pred).values())
    return FieldCounts(tp=tp, fp=sum(pred.values()) - tp, fn=sum(gold.values()) - tp)


def score(example: ReceiptExample, prediction: Prediction) -> ExampleScore:
    if example.example_id != prediction.example_id:
        raise ValueError(f"scoring {prediction.example_id} against {example.example_id}")
    parsed = prediction.parsed or {}
    gold, pred = _pairs(example.target, normalize.strict), _pairs(parsed, normalize.strict)
    gold_lenient, pred_lenient = _pairs(example.target, normalize.lenient), _pairs(parsed, normalize.lenient)
    return ExampleScore(
        example_id=example.example_id,
        variant=prediction.variant,
        valid_json=prediction.parsed is not None,
        exact_match=prediction.parsed is not None and gold == pred,
        overall=_counts(gold, pred),
        numeric=_counts(_only(gold, "numeric"), _only(pred, "numeric")),
        text=_counts(_only(gold, "text"), _only(pred, "text")),
        numeric_lenient=_counts(_only(gold_lenient, "numeric"), _only(pred_lenient, "numeric")),
        ted_accuracy=ted_accuracy(prediction.parsed, example.target),
    )


def total(counts: Iterable[FieldCounts | None]) -> FieldCounts:
    tp = fp = fn = 0
    for c in counts:
        if c is not None:
            tp, fp, fn = tp + c.tp, fp + c.fp, fn + c.fn
    return FieldCounts(tp=tp, fp=fp, fn=fn)


def f1(c: FieldCounts) -> float:
    denom = 2 * c.tp + c.fp + c.fn
    return 2 * c.tp / denom if denom else 1.0  # nothing to find and nothing claimed


def precision(c: FieldCounts) -> float:
    return c.tp / (c.tp + c.fp) if c.tp + c.fp else 1.0


def recall(c: FieldCounts) -> float:
    return c.tp / (c.tp + c.fn) if c.tp + c.fn else 1.0


def _field(stat: Callable[[FieldCounts], float], group: str) -> Metric:
    return lambda scores: stat(total(getattr(s, group) for s in scores))


def _rate(flag: str) -> Metric:
    return lambda scores: sum(getattr(s, flag) for s in scores) / len(scores)


def _mean_ted(scores: Sequence[ExampleScore]) -> float:
    return sum(s.ted_accuracy or 0.0 for s in scores) / len(scores)


METRICS: dict[str, Metric] = {
    "f1": _field(f1, "overall"),
    "f1_numeric": _field(f1, "numeric"),
    "f1_text": _field(f1, "text"),
    "f1_numeric_lenient": _field(f1, "numeric_lenient"),
    "precision": _field(precision, "overall"),
    "recall": _field(recall, "overall"),
    "valid_json": _rate("valid_json"),
    "exact_match": _rate("exact_match"),
    "ted_accuracy": _mean_ted,
}


def summarize(
    scores: Sequence[ExampleScore], n_boot: int = 2000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Every metric as {"value", "lo", "hi"}: the point estimate and its 95% percentile bootstrap interval."""
    out = {}
    for name, metric in METRICS.items():
        value, lo, hi = interval(scores, metric, n_boot=n_boot, seed=seed)
        out[name] = {"value": value, "lo": lo, "hi": hi}
    return out
