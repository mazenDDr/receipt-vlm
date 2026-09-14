"""How a receipt JSON becomes (key path, value) pairs, and which keys hold numbers vs. text."""

from __future__ import annotations

from typing import Any, Literal

Kind = Literal["numeric", "text"]

# Leaf key paths of the CORD v2 label set (Park et al., 2019), list indices dropped.
# Numeric keys hold prices, counts and item codes: the fields where one wrong digit changes the meaning.
TEXT_KEYS = frozenset({"menu.nm", "menu.sub.nm", "menu.etc", "menu.sub.etc", "menu.vatyn", "void_menu.nm"})
NUMERIC_KEYS = frozenset(
    {
        "menu.num",
        "menu.unitprice",
        "menu.cnt",
        "menu.discountprice",
        "menu.price",
        "menu.itemsubtotal",
        "menu.sub.num",
        "menu.sub.unitprice",
        "menu.sub.cnt",
        "menu.sub.discountprice",
        "menu.sub.price",
        "menu.sub.itemsubtotal",
        "void_menu.price",
        "sub_total.subtotal_price",
        "sub_total.discount_price",
        "sub_total.service_price",
        "sub_total.othersvc_price",
        "sub_total.tax_price",
        "sub_total.etc",
        "total.total_price",
        "total.total_etc",
        "total.cashprice",
        "total.changeprice",
        "total.creditcardprice",
        "total.emoneyprice",
        "total.menutype_cnt",
        "total.menuqty_cnt",
    }
)


def kind(path: str) -> Kind:
    if path in TEXT_KEYS:
        return "text"
    if path in NUMERIC_KEYS:
        return "numeric"
    # Keys outside the label set only come from model output (so they are false positives); go by the name.
    last = path.rsplit(".", 1)[-1]
    return "numeric" if "price" in last or "cnt" in last or last == "num" else "text"


def flatten(data: Any, path: str = "") -> list[tuple[str, str]]:
    """(key path, value) per leaf. List items share their parent's path, so item order doesn't matter."""
    if isinstance(data, dict):
        return [
            pair for key, value in data.items() for pair in flatten(value, f"{path}.{key}" if path else key)
        ]
    if isinstance(data, list):
        return [pair for item in data for pair in flatten(item, path)]
    if data is None:
        return []
    return [(path, str(data))]
