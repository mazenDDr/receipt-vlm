"""Fine-tune Qwen2.5-VL on receipts: QLoRA (nf4 base), bf16 LoRA, or partial full fine-tuning.

Dev receipts are scored by generation (the task metric, not loss) after every epoch and on all of dev at
the end, through the same runner and metrics as every other variant.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from receipt_vlm.data.build import load_examples
from receipt_vlm.eval import report
from receipt_vlm.infer.runner import InferConfig, bnb_skip_modules, check_vision_precision, predict
from receipt_vlm.schemas import ReceiptExample
from receipt_vlm.train.collate import Collator, ExampleDataset

# Anchored to the language model: the vision blocks reuse the names gate_proj/up_proj/down_proj, so a plain
# name list would also put adapters into 96 vision layers.
LM_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
MERGER_TARGETS = r".*visual\.merger\.mlp\.(0|2)"


class TrainConfig(BaseModel):
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    variant: str = "qlora-r16-lm"  # models/<variant>/ and the W&B run name
    examples_path: str = "data/processed/examples.jsonl"
    precision: Literal["nf4", "bf16"] = "nf4"  # nf4 = QLoRA
    quantize_vision: bool = False
    method: Literal["lora", "partial"] = "lora"
    targets: Literal["lm", "lm+merger"] = "lm"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # method=partial: train the merger + the last N language layers in full, no adapters
    partial_last_n_layers: int = 4
    learning_rate: float = 2e-4
    epochs: float = 2.0
    batch_size: int = 1
    grad_accum: int = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    optim: str = "adamw_torch"
    min_pixels: int = 56 * 56
    max_pixels: int = 1024 * 28 * 28
    max_new_tokens: int = 1024
    train_limit: int | None = None  # smoke tests only
    eval_limit: int | None = 30  # dev receipts scored after each epoch
    final_eval_limit: int | None = None  # dev receipts scored at the end (None = all); smoke tests use a few
    num_workers: int = 4
    seed: int = 0
    report_to: Literal["wandb", "none"] = "wandb"
    # the user's W&B workspace (sole member); the free tier has no personal entity and can't create teams
    wandb_entity: str = "khaledmazen456-zewail-city-of-science-and-technology"
    wandb_project: str = "receipt-vlm"
    # offline: saved on disk, uploaded later with `wandb sync`
    wandb_mode: Literal["online", "offline"] = "online"
    # e.g. "expandable_segments:True" for arms near the memory limit (bf16 LoRA, partial fine-tuning)
    cuda_alloc_conf: str | None = None


def apply_cuda_alloc_conf(value: str | None) -> None:
    """Set PYTORCH_CUDA_ALLOC_CONF. It only works if set before torch starts, so refuse once it's too late."""
    import sys

    if not value:
        return
    if "torch" in sys.modules:
        raise RuntimeError(
            "PYTORCH_CUDA_ALLOC_CONF must be set before torch is imported; it would be ignored"
        )
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = value


def lora_target_regex(targets: str) -> str:
    return LM_TARGETS if targets == "lm" else f"{LM_TARGETS}|{MERGER_TARGETS}"


def expected_wrapped(n_layers: int, targets: str) -> int:
    return n_layers * 7 + (2 if targets == "lm+merger" else 0)


def count_wrapped(model: Any) -> int:
    return sum(
        hasattr(m, "lora_A") and not n.endswith(("lora_A", "lora_B")) for n, m in model.named_modules()
    )


def build_model(cfg: TrainConfig) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if cfg.method == "partial" and cfg.precision != "bf16":
        raise ValueError("partial fine-tuning updates real weights; use precision bf16")
    quantization = None
    if cfg.precision == "nf4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=bnb_skip_modules(cfg.quantize_vision),
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        quantization_config=quantization,
    )
    if cfg.precision == "nf4":
        check_vision_precision(model, cfg.quantize_vision)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = len(model.model.language_model.layers)
    if cfg.method == "lora":
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=lora_target_regex(cfg.targets),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        wrapped, expected = count_wrapped(model), expected_wrapped(n_layers, cfg.targets)
        if wrapped != expected:
            raise RuntimeError(
                f"LoRA wrapped {wrapped} modules, expected {expected} for targets={cfg.targets}"
            )
    else:
        trained = [
            model.model.visual.merger,
            *model.model.language_model.layers[-cfg.partial_last_n_layers :],
        ]
        trained.append(model.model.language_model.norm)
        for module in trained:
            module.float()  # fp32 master weights for the parts that learn; autocast still runs them in bf16
            for p in module.parameters():
                p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(cfg.model)
    return model, processor


def evaluate(
    model: Any, processor: Any, examples: list[ReceiptExample], cfg: TrainConfig, out_dir: Path
) -> dict:
    """Generate on dev receipts with the in-memory model and score them like any other run."""
    infer_cfg = InferConfig(
        model=cfg.model,
        variant=cfg.variant,
        split="dev",
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
        max_new_tokens=cfg.max_new_tokens,
    )
    was_training = model.training
    model.eval()
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w") as f:
        for pred in predict(examples, infer_cfg, model, processor):
            f.write(pred.model_dump_json() + "\n")
    summary = report.write(out_dir, examples, cfg.variant, "dev")
    model.config.use_cache = False
    processor.tokenizer.padding_side = "right"
    if was_training:
        model.train()
    return summary


def _log_dev(summary: dict, step: int, prefix: str = "dev") -> None:
    import wandb

    if wandb.run is None:
        return
    metrics = {f"{prefix}/{k}": v["value"] for k, v in summary["metrics"].items()}
    metrics[f"{prefix}/truncated"] = summary["run"]["truncated"]
    wandb.log(metrics, step=step)


def _sample_table(out_dir: Path, examples: list[ReceiptExample], step: int, n: int = 5) -> None:
    import wandb

    if wandb.run is None:
        return
    preds = report.load_predictions(out_dir / "predictions.jsonl")
    table = wandb.Table(columns=["example_id", "gold", "prediction"])
    for e in examples[:n]:
        table.add_data(e.example_id, json.dumps(e.target, ensure_ascii=False), preds[e.example_id].raw_output)
    wandb.log({"dev/samples": table}, step=step)


def train(cfg: TrainConfig, run_dir: Path, model_dir: Path) -> dict:
    import math

    import torch
    import wandb
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    set_seed(cfg.seed)
    examples = load_examples(cfg.examples_path)
    train_set = [e for e in examples if e.split == "train"][: cfg.train_limit]
    dev = [e for e in examples if e.split == "dev"]
    model, processor = build_model(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"{cfg.variant}: {len(train_set)} train receipts, {trainable / 1e6:.1f}M trainable parameters",
        flush=True,
    )

    if cfg.report_to == "wandb":
        os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"] = cfg.wandb_entity, cfg.wandb_project
        os.environ["WANDB_MODE"] = cfg.wandb_mode  # the Trainer's W&B callback reads it too
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=cfg.variant,
            mode=cfg.wandb_mode,
            config={**cfg.model_dump(), "trainable_params": trainable},
            dir=str(run_dir),
        )

    train_peaks_mb: list[float] = []

    class DevEval(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
            # read the training peak before generation resets the counter
            train_peaks_mb.append(torch.cuda.max_memory_allocated() / 2**20)
            out = run_dir / f"epoch{state.epoch:.2f}"
            summary = evaluate(model, processor, dev[: cfg.eval_limit], cfg, out)
            print(f"epoch {state.epoch:.2f}: dev f1 {summary['metrics']['f1']['value']:.3f}", flush=True)
            _log_dev(summary, state.global_step)
            _sample_table(out, dev, state.global_step)
            torch.cuda.reset_peak_memory_stats()

    steps_per_epoch = math.ceil(len(train_set) / (cfg.batch_size * cfg.grad_accum))
    warmup_steps = max(1, round(math.ceil(steps_per_epoch * cfg.epochs) * cfg.warmup_ratio))
    args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_steps=warmup_steps,  # integer steps: warmup_ratio is deprecated in transformers 5.x
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="no",
        report_to=[] if cfg.report_to == "none" else ["wandb"],
        run_name=cfg.variant,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=cfg.num_workers,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ExampleDataset(train_set),
        data_collator=Collator(processor, cfg.min_pixels, cfg.max_pixels),
        callbacks=[DevEval()],
    )
    resume = any((model_dir / "checkpoints").glob("checkpoint-*"))
    t0 = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    train_s = time.perf_counter() - t0
    model.save_pretrained(model_dir / ("adapter" if cfg.method == "lora" else "full"))
    (run_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    facts = {
        "train_seconds": train_s,
        "train_loss": result.training_loss,
        "train_peak_vram_mb": max(train_peaks_mb, default=0.0),
        "trainable_params": trainable,
    }
    # Saved before the final eval: a job that dies there (e.g. the machine sleeps) keeps its training facts,
    # and the eval can be rerun alone with scripts/run_infer.py --adapter.
    (run_dir / "train_summary.json").write_text(json.dumps(facts, indent=2))

    # final scores at the run dir's top level, like inference runs
    final = evaluate(model, processor, dev[: cfg.final_eval_limit], cfg, run_dir)
    _log_dev(final, trainer.state.global_step, prefix="final_dev")
    if wandb.run is not None:
        wandb.summary.update(facts)
        wandb.finish()
    return {**facts, **final}
