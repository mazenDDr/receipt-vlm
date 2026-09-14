"""Tree-edit-distance accuracy, as in Donut's CORD evaluation (Kim et al., 2022), so numbers are comparable.

accuracy = max(0, 1 - TED(prediction, gold) / TED(empty, gold)). Keys are inner nodes, list items are
<subtree> nodes, values are leaves; changing a leaf costs its character edit distance. The tree is ordered,
so items listed in a different order cost TED (field F1 ignores order).
"""

from __future__ import annotations

from typing import Any

import zss

_LEAF = "<leaf>"


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _normalize(data: Any) -> Any:
    """Donut's normalize_dict: sort keys, wrap single values in lists, drop empty values."""
    if isinstance(data, dict):
        out = {}
        for key in sorted(data, key=lambda k: (len(k), k)):
            value = _normalize(data[key])
            if value:
                out[key] = value if isinstance(value, list) else [value]
        return out
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            return [n for n in (_normalize(item) for item in data) if n]
        return [
            str(item).strip() for item in data if isinstance(item, str | int | float) and str(item).strip()
        ]
    text = "" if data is None else str(data).strip()
    return [text] if text else []


def _tree(data: Any, name: str = "<root>") -> zss.Node:
    node = zss.Node(name)
    if isinstance(data, dict):
        for key, value in data.items():
            node.addkid(_tree(value, key))
    elif all(isinstance(item, dict) for item in data):
        for item in data:
            node.addkid(_tree(item, "<subtree>"))
    else:
        for item in data:
            node.addkid(zss.Node(f"{_LEAF}{item}"))
    return node


def _insert_remove_cost(node: zss.Node) -> int:
    return len(node.label) - len(_LEAF) if node.label.startswith(_LEAF) else 1


def _update_cost(a: zss.Node, b: zss.Node) -> int:
    a_leaf, b_leaf = a.label.startswith(_LEAF), b.label.startswith(_LEAF)
    if a_leaf and b_leaf:
        return _levenshtein(a.label[len(_LEAF) :], b.label[len(_LEAF) :])
    if a_leaf != b_leaf:
        leaf = a if a_leaf else b
        return 1 + len(leaf.label) - len(_LEAF)
    return int(a.label != b.label)


def _distance(a: zss.Node, b: zss.Node) -> float:
    return zss.distance(
        a,
        b,
        get_children=zss.Node.get_children,
        insert_cost=_insert_remove_cost,
        remove_cost=_insert_remove_cost,
        update_cost=_update_cost,
    )


def ted_accuracy(prediction: dict[str, Any] | None, gold: dict[str, Any]) -> float:
    gold_tree = _tree(_normalize(gold))
    worst = _distance(_tree(_normalize({})), gold_tree)
    if worst == 0:
        return 1.0
    return max(0.0, 1 - _distance(_tree(_normalize(prediction or {})), gold_tree) / worst)
