"""YAML configs loaded into pydantic models. Every run saves its resolved config next to its outputs."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def load(path: str | Path, model: type[M]) -> M:
    return model.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def save(cfg: BaseModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
