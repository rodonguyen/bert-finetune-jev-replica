"""Run two LoRA training steps to check the GLiNER 2.5 CUDA training path."""

import json
import random
from pathlib import Path

import torch
from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

ROOT = Path(__file__).parent
OUTPUT = ROOT / "runs" / "pilot_lora"
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this PyTorch environment")

with (ROOT / "data" / "development_train.jsonl").open(encoding="utf-8") as source:
    rows = [json.loads(line) for line in source]
examples = random.Random(42).sample(rows, 8)

model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1", map_location="cuda")
config = TrainingConfig(
    output_dir=str(OUTPUT),
    num_epochs=1,
    max_steps=2,
    batch_size=1,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_target_modules=["encoder"],
    save_adapter_only=True,
    task_lr=5e-4,
    fp16=True,
    eval_strategy="no",
    save_best=False,
    num_workers=0,
    pin_memory=False,
    fused_optimizer=False,
    seed=42,
)
trainer = ExtractorTrainer(model, config)
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")

result = trainer.train(train_data=examples)
adapter = OUTPUT / "final" / "adapter_model.safetensors"
print(f"Completed {result['total_steps']} update steps in {result['total_time_seconds']:.1f}s")
print(f"Saved adapter: {adapter}")
