"""Train three LoRA epochs on BANKING77 development data, selecting by validation loss."""

import json
from pathlib import Path

import torch
from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUTPUT = ROOT / "runs" / "development_lora"

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this PyTorch environment")
split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
if split["counts"] != {"development_train": 9000, "validation": 1003}:
    raise SystemExit("Unexpected development split; inspect data/split.json first")
if OUTPUT.exists():
    raise SystemExit(f"Output already exists: {OUTPUT}. Keep it or move it before a new run.")

model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1", map_location="cuda")
config = TrainingConfig(
    output_dir=str(OUTPUT),
    num_epochs=3,
    batch_size=1,
    gradient_accumulation_steps=4,
    eval_batch_size=1,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    lora_target_modules=["encoder", "classifier"],
    save_adapter_only=True,
    task_lr=5e-4,
    weight_decay=0.01,
    scheduler_type="linear",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    fp16=True,
    eval_strategy="epoch",
    save_best=True,
    metric_for_best="eval_loss",
    logging_steps=100,
    num_workers=0,
    pin_memory=False,
    fused_optimizer=False,
    seed=42,
)
trainer = ExtractorTrainer(model, config)
print("Training on 9,000 development rows for three epochs (micro-batch 1, effective batch 4).")
print("Validation loss is checked after each epoch; official test rows are excluded.")
result = trainer.train(
    train_data=str(DATA / "development_train.jsonl"),
    eval_data=str(DATA / "validation.jsonl"),
)
adapter = OUTPUT / "best" / "adapter_model.safetensors"
if result["total_steps"] != 6750 or not adapter.exists():
    raise RuntimeError("Expected 6,750 optimizer updates and a saved best adapter")

summary = {
    "checkpoint": "fastino/gliner2.5-base-v1",
    "split_seed": split["seed"],
    "training_examples": 9000,
    "epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 4,
    "optimizer_steps": result["total_steps"],
    "training_seconds": result["total_time_seconds"],
    "best_validation_loss": result["best_metric"],
    "validation_history": result["eval_metrics_history"],
    "adapter": str(adapter),
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"Completed {result['total_steps']} updates in {result['total_time_seconds'] / 60:.1f} minutes")
print(f"Best validation loss: {result['best_metric']:.4f}")
print(f"Saved best adapter: {adapter}")
