"""Compare untouched and trained GLiNER on all 1,003 validation rows."""

import json
from pathlib import Path

import torch
from gliner2 import AutoExtractor
from peft import PeftModel

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ADAPTER = ROOT / "runs" / "development_lora" / "best"
OUTPUT = ROOT / "runs" / "development_validation.json"

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this PyTorch environment")
if not (ADAPTER / "adapter_model.safetensors").exists():
    raise SystemExit("Run train_development.py first to save the adapter")

labels = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["labels"]
with (DATA / "validation.jsonl").open(encoding="utf-8") as source:
    examples = [json.loads(line) for line in source]
if len(examples) != 1003 or len(labels) != 77:
    raise SystemExit("Unexpected validation size or label count")

model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1", map_location="cuda")
model.eval()


def predict(extractor):
    predictions = []
    with torch.inference_mode():
        for row in examples:
            predictions.append(
                extractor.classify_text(row["input"], {"banking_intent": labels})["banking_intent"]
            )
    return predictions


base_predictions = predict(model)
adapted = PeftModel.from_pretrained(model, str(ADAPTER)).to("cuda")
adapted.eval()
trained_predictions = predict(adapted)

paired = []
for row, base, trained in zip(examples, base_predictions, trained_predictions):
    gold = row["output"]["classifications"][0]["true_label"][0]
    paired.append({"text": row["input"], "gold": gold, "base": base, "trained": trained})


def metrics(key):
    accuracy = sum(item[key] == item["gold"] for item in paired) / len(paired)
    f1_values = []
    for label in labels:
        true_positive = sum(item[key] == label and item["gold"] == label for item in paired)
        false_positive = sum(item[key] == label and item["gold"] != label for item in paired)
        false_negative = sum(item[key] != label and item["gold"] == label for item in paired)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return {"accuracy": accuracy, "macro_f1": sum(f1_values) / len(f1_values)}


result = {
    "split": "validation",
    "checkpoint": "fastino/gliner2.5-base-v1",
    "adapter": str(ADAPTER),
    "examples": len(paired),
    "labels": len(labels),
    "base": metrics("base"),
    "trained": metrics("trained"),
    "predictions": paired,
}
OUTPUT.parent.mkdir(exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Untouched: accuracy {result['base']['accuracy']:.1%}, macro-F1 {result['base']['macro_f1']:.3f}")
print(f"Trained:   accuracy {result['trained']['accuracy']:.1%}, macro-F1 {result['trained']['macro_f1']:.3f}")
print(f"Accuracy change: {(result['trained']['accuracy'] - result['base']['accuracy']) * 100:+.1f} percentage points")
print(f"Saved paired predictions: {OUTPUT}")
