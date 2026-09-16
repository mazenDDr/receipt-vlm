"""Run one model variant over one split, then score it.

    python scripts/run_infer.py --config configs/infer.yaml --split dev [--max-pixels 200704] [--limit 5]
    python scripts/run_infer.py --resume outputs/runs/<run_id>      # finish an interrupted run
    <vllm env>/bin/python scripts/run_infer.py --backend vllm --model models/<awq checkpoint> --split dev

Predictions are appended as they are produced, so a crash loses at most the receipt in progress.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

from receipt_vlm import config
from receipt_vlm.data.build import load_examples
from receipt_vlm.eval import report
from receipt_vlm.infer import backends, runner

OVERRIDES = (
    "model",
    "backend",
    "variant",
    "split",
    "offset",
    "limit",
    "max_pixels",
    "max_new_tokens",
    "max_model_len",
    "repetition_penalty",
    "precision",
    "adapter",
    "mmproj",
    "llama_server",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--resume", default=None, help="existing run dir; its saved config is used")
    parser.add_argument("--tag", default="", help="suffix for the run dir, e.g. px256")
    parser.add_argument("--model", help="HF id or a local checkpoint dir, e.g. an AWQ-quantized model")
    parser.add_argument(
        "--backend",
        choices=["hf", "vllm", "llamacpp"],
        help="engine: transformers, vLLM (AWQ W4A16) or llama.cpp (GGUF)",
    )
    parser.add_argument("--mmproj", help="llama.cpp: the vision projector GGUF")
    parser.add_argument("--llama-server", help="llama.cpp: path to the llama-server binary")
    parser.add_argument("--variant")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, help="start at this receipt, to separate position from content")
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-model-len", type=int, help="context size (vLLM --max-model-len, llama.cpp -c)")
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--precision")
    parser.add_argument("--adapter")
    args = parser.parse_args()

    if args.resume:
        run_dir = Path(args.resume)
        cfg = config.load(run_dir / "config.yaml", runner.InferConfig)
    else:
        cfg = config.load(args.config, runner.InferConfig)
        updates = {k: getattr(args, k) for k in OVERRIDES if getattr(args, k) is not None}
        cfg = runner.InferConfig.model_validate({**cfg.model_dump(), **updates})
        tag = f"_{args.tag}" if args.tag else ""
        run_dir = Path("outputs/runs") / f"{time.strftime('%Y%m%d-%H%M')}_{cfg.variant}_{cfg.split}{tag}"
        config.save(cfg, run_dir / "config.yaml")

    in_split = [e for e in load_examples(cfg.examples_path) if e.split == cfg.split][cfg.offset :]
    examples = in_split[: cfg.limit]
    pred_path = run_dir / "predictions.jsonl"
    done = report.load_predictions(pred_path)
    todo = [e for e in examples if e.example_id not in done]
    print(f"{run_dir}: {len(todo)} receipts to run, {len(done)} already done ({cfg.backend})", flush=True)

    if todo:
        engine = backends.get(cfg.backend)
        model, processor = engine.load(cfg)
        with pred_path.open("a") as f:
            for i, pred in enumerate(engine.predict(todo, cfg, model, processor), 1):
                f.write(pred.model_dump_json() + "\n")
                f.flush()
                print(
                    f"[{i}/{len(todo)}] {pred.example_id} out={pred.output_tokens} "
                    f"{pred.latency_s:.1f}s valid={pred.parsed is not None} trunc={pred.truncated}",
                    flush=True,
                )
        del model
        gc.collect()

    summary = report.write(run_dir, examples, cfg.variant, cfg.split)
    print(report.render(summary))


if __name__ == "__main__":
    main()
