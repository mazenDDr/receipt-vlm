"""Train every arm of an ablation stage, one after another, each in a fresh process (one GPU job at a time).

    python scripts/sweep.py configs/sweeps/stage-a-lr.yaml

Arms already trained (models/<variant>/adapter or full/) are skipped, so a stopped sweep can simply be rerun.
Each arm's resolved config is saved under outputs/sweeps/<stage>/ before it starts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from receipt_vlm import config
from receipt_vlm.train import sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", help="a sweep file under configs/sweeps/")
    args = parser.parse_args()
    arms = sweep.load_arms(args.sweep)
    out_dir = Path("outputs/sweeps") / Path(args.sweep).stem
    for cfg in arms:
        if sweep.is_trained(cfg.variant):
            print(f"skip {cfg.variant}: already trained", flush=True)
            continue
        path = out_dir / f"{cfg.variant}.yaml"
        config.save(cfg, path)
        print(f"=== {time.strftime('%F %T')} start {cfg.variant}", flush=True)
        code = subprocess.call([sys.executable, "scripts/train.py", "--config", str(path)])
        print(f"=== {time.strftime('%F %T')} end {cfg.variant} (exit {code})", flush=True)


if __name__ == "__main__":
    main()
