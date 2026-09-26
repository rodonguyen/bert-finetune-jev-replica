"""Compare the untouched model and two-step LoRA pilot on 20 validation rows."""

import json
import random
from pathlib import Path

import torch
from gliner2 import AutoExtractor
from peft import PeftModel

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ADAPTER = ROOT / "runs" / "pilot_lora" / "final"
OUTPUT = ROOT / "runs" / "pilot_comparison.json"

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this PyTorch environment")
if not (ADAPTER / "adapter_model.safetensors").exists():
    raise SystemExit("Run train_pilot.py first to save the adapter")

labels = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["labels"]
with (DATA / "validation.jsonl").open(encoding="utf-8") as source:
    rows = [json.loads(line) for line in source]
examples = random.Random(42).sample(rows, 20)

model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1", map_location="cuda")
model.eval()


def predict(extractor):
    with torch.inference_mode():
        return [
            extractor.classify_text(row["input"], {"banking_intent": labels})["banking_intent"]
            for row in examples
        ]


base_predictions = predict(model)
adapted = PeftModel.from_pretrained(model, str(ADAPTER)).to("cuda")
adapted.eval()
adapted_predictions = predict(adapted)

paired = []
for row, base, tuned in zip(examples, base_predictions, adapted_predictions):
    gold = row["output"]["classifications"][0]["true_label"][0]
    paired.append({"text": row["input"], "gold": gold, "base": base, "pilot": tuned})

base_correct = sum(item["base"] == item["gold"] for item in paired)
pilot_correct = sum(item["pilot"] == item["gold"] for item in paired)
changed = [item for item in paired if item["base"] != item["pilot"]]
result = {
    "split": "validation",
    "sample_seed": 42,
    "checkpoint": "fastino/gliner2.5-base-v1",
    "adapter": str(ADAPTER),
    "label_count": len(labels),
    "sample_count": len(paired),
    "base_correct": base_correct,
    "pilot_correct": pilot_correct,
    "changed_count": len(changed),
    "predictions": paired,
}
OUTPUT.parent.mkdir(exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Untouched: {base_correct}/{len(paired)}; two-step LoRA: {pilot_correct}/{len(paired)}")
print(f"Changed predictions: {len(changed)}/{len(paired)}")
for item in changed:
    print(f"  {item['text']!r}: {item['base']} -> {item['pilot']} (gold: {item['gold']})")
print(f"Saved paired predictions: {OUTPUT}")
