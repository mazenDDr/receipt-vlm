"""Ablation stages: a base training config plus arms that override it, trained one after another."""

from __future__ import annotations

from pathlib import Path

import yaml

from receipt_vlm import config
from receipt_vlm.train.qlora import TrainConfig


def load_arms(sweep_path: str | Path) -> list[TrainConfig]:
    """Every arm as a full config. Unknown keys are an error: pydantic would silently ignore a typo."""
    spec = yaml.safe_load(Path(sweep_path).read_text())
    base = config.load(spec["base"], TrainConfig).model_dump()
    arms: list[TrainConfig] = []
    for arm in spec["arms"]:
        unknown = set(arm) - set(TrainConfig.model_fields)
        if unknown:
            raise ValueError(f"unknown settings {sorted(unknown)} in arm {arm}")
        if "variant" not in arm:
            raise ValueError(f"every arm needs its own variant name: {arm}")
        cfg = TrainConfig.model_validate({**base, **arm})
        if any(a.variant == cfg.variant for a in arms):
            raise ValueError(f"variant {cfg.variant} appears twice")
        arms.append(cfg)
    return arms


def is_trained(variant: str, models_dir: Path = Path("models")) -> bool:
    return any((models_dir / variant / kind).exists() for kind in ("adapter", "full"))
