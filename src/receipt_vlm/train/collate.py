"""Training batches: the inference prompt + image + target JSON, with the loss only on the answer tokens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from receipt_vlm.infer.runner import check_image_cap, image_kwargs
from receipt_vlm.prompts import build_messages
from receipt_vlm.schemas import ReceiptExample

IGNORE = -100
ASSISTANT_HEADER = "<|im_start|>assistant\n"


def answer_labels(
    input_ids: Sequence[int], attention_mask: Sequence[int], header: Sequence[int]
) -> list[int]:
    """Labels for the answer only: everything up to the last assistant header, and padding, is ignored."""
    header = list(header)
    for i in range(len(input_ids) - len(header), -1, -1):
        if list(input_ids[i : i + len(header)]) == header:
            start = i + len(header)
            break
    else:
        raise ValueError("no assistant header in the sequence")
    return [
        t if i >= start and m else IGNORE
        for i, (t, m) in enumerate(zip(input_ids, attention_mask, strict=True))
    ]


class ExampleDataset:
    """A plain sequence of receipts; the collator loads images, so workers only pickle small objects."""

    def __init__(self, examples: Sequence[ReceiptExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> ReceiptExample:
        return self.examples[i]


class Collator:
    def __init__(self, processor: Any, min_pixels: int, max_pixels: int) -> None:
        self.processor = processor
        self.max_pixels = max_pixels
        self.images_kwargs = image_kwargs(min_pixels, max_pixels)
        self.header = processor.tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)

    def __call__(self, examples: Sequence[ReceiptExample]) -> dict[str, Any]:
        import torch
        from PIL import Image

        texts, images = [], []
        for e in examples:
            texts.append(self.processor.apply_chat_template(build_messages(e.target), tokenize=False))
            with Image.open(e.image_path) as image:
                images.append(image.convert("RGB"))
        self.processor.tokenizer.padding_side = "right"
        batch = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt", images_kwargs=self.images_kwargs
        )
        check_image_cap(
            batch["image_grid_thw"].tolist(), self.processor.image_processor.patch_size, self.max_pixels
        )
        batch["labels"] = torch.tensor(
            [
                answer_labels(ids, mask, self.header)
                for ids, mask in zip(
                    batch["input_ids"].tolist(), batch["attention_mask"].tolist(), strict=True
                )
            ]
        )
        return batch
