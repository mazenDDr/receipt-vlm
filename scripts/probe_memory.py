"""Peak GPU memory of real training steps on the longest train receipts, before committing to a full run.

    python scripts/probe_memory.py                                   # the QLoRA base config
    python scripts/probe_memory.py --precision bf16                  # bf16 LoRA
    python scripts/probe_memory.py --precision bf16 --method partial --optim paged_adamw_8bit

The model is built exactly as training builds it and the optimizer step is included, so the peak covers
weights, adapters, activations, gradients and optimizer state. Run it only when no other job is on the GPU.
"""

from __future__ import annotations

import argparse

from receipt_vlm import config
from receipt_vlm.data.build import load_examples
from receipt_vlm.train import qlora
from receipt_vlm.train.collate import Collator

OVERRIDES = {"precision": str, "method": str, "targets": str, "lora_r": int, "optim": str, "max_pixels": int}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/qlora-r16-lm.yaml")
    parser.add_argument("--n", type=int, default=3, help="how many of the longest receipts to step on")
    for name, kind in OVERRIDES.items():
        parser.add_argument("--" + name.replace("_", "-"), type=kind)
    args = parser.parse_args()
    cfg = config.load(args.config, qlora.TrainConfig)
    updates = {k: getattr(args, k) for k in OVERRIDES if getattr(args, k) is not None}
    cfg = qlora.TrainConfig.model_validate({**cfg.model_dump(), **updates})

    import torch

    model, processor = qlora.build_model(cfg)
    model.train()
    collate = Collator(processor, cfg.min_pixels, cfg.max_pixels)
    train = [e for e in load_examples(cfg.examples_path) if e.split == "train"]
    by_length = sorted(train, key=lambda e: int(collate([e])["attention_mask"].sum()))
    longest = by_length[-args.n :]

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optim == "paged_adamw_8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(params, lr=1e-5)
    else:
        optimizer = torch.optim.AdamW(params, lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    for example in longest:
        batch = collate([example]).to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    seq = int(collate([longest[-1]])["attention_mask"].sum())
    trainable = sum(p.numel() for p in params)
    print(
        f"{cfg.precision} {cfg.method} targets={cfg.targets} r={cfg.lora_r} optim={cfg.optim}: "
        f"{trainable / 1e6:.1f}M trainable, longest sequence {seq} tokens, "
        f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
        f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB "
        f"(card {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
