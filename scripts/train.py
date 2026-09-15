"""Fine-tune one variant, then score it on all of dev.

    python scripts/train.py --config configs/train/qlora-r16-lm.yaml
    python scripts/train.py --config configs/train/qlora-r16-lm.yaml --variant smoke --train-limit 16 \
        --epochs 1 --eval-limit 2 --report-to none                      # a quick end-to-end check

Writes models/<variant>/ (checkpoints, then adapter/ or full/) and outputs/runs/<time>_<variant>_train/
(config, log history, per-epoch dev predictions, final dev summary.json + report.md).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from receipt_vlm import config
from receipt_vlm.train import qlora

OVERRIDES = {
    "variant": str,
    "epochs": float,
    "learning_rate": float,
    "lora_r": int,
    "lora_alpha": int,
    "targets": str,
    "precision": str,
    "method": str,
    "partial_last_n_layers": int,
    "max_pixels": int,
    "train_limit": int,
    "eval_limit": int,
    "final_eval_limit": int,
    "grad_accum": int,
    "report_to": str,
    "seed": int,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/qlora-r16-lm.yaml")
    for name, kind in OVERRIDES.items():
        parser.add_argument("--" + name.replace("_", "-"), type=kind)
    args = parser.parse_args()
    cfg = config.load(args.config, qlora.TrainConfig)
    updates = {k: getattr(args, k) for k in OVERRIDES if getattr(args, k) is not None}
    cfg = qlora.TrainConfig.model_validate({**cfg.model_dump(), **updates})
    qlora.apply_cuda_alloc_conf(cfg.cuda_alloc_conf)  # before anything imports torch

    model_dir = Path("models") / cfg.variant
    if (model_dir / "adapter").exists() or (model_dir / "full").exists():
        raise SystemExit(
            f"{model_dir} already holds a trained model; models are write-once, pick a new --variant"
        )
    run_dir = Path("outputs/runs") / f"{time.strftime('%Y%m%d-%H%M')}_{cfg.variant}_train"
    config.save(cfg, run_dir / "config.yaml")

    summary = qlora.train(cfg, run_dir, model_dir)
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"trained in {summary['train_seconds'] / 60:.1f} min, loss {summary['train_loss']:.4f}, "
        f"training peak VRAM {summary['train_peak_vram_mb']:.0f} MB"
    )
    print((run_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
