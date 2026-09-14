"""Value normalization. Strict is the reported metric; lenient is a diagnostic for numeric fields."""

from __future__ import annotations

import re

_SPACE = re.compile(r"\s+")
_ZERO_CENTS = re.compile(r"[.,]0{1,2}$")


def strict(value: str) -> str:
    """Strip only, as in Donut's CORD evaluation, so one wrong character makes the field wrong."""
    return value.strip()


def lenient(value: str) -> str:
    """Forgive formatting, not reading: 'Rp 16.500', '@16,500' and '16500' all become '16500'.

    Numbers keep only their digits (and a leading minus); zero cents ('70000.00') are dropped first.
    Values without digits are casefolded with whitespace collapsed.
    """
    value = _SPACE.sub(" ", value).strip().casefold()
    if not any(ch.isdigit() for ch in value):
        return value
    sign = "-" if value.lstrip("rp@ ").startswith("-") else ""
    return sign + "".join(ch for ch in _ZERO_CENTS.sub("", value) if ch.isdigit())
