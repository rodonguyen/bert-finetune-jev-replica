"""Ask an untouched GLiNER 2.5 model to classify one BANKING77 message."""

import json
import time
from pathlib import Path

from gliner2 import AutoExtractor

MODEL = "fastino/gliner2.5-base-v1"
DATA = Path(__file__).parent / "data"

manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
with (DATA / "train.jsonl").open(encoding="utf-8") as source:
    example = json.loads(source.readline())

text = example["input"]
labels = manifest["labels"]

started = time.perf_counter()
model = AutoExtractor.from_pretrained(MODEL, map_location="cpu")
load_seconds = time.perf_counter() - started

# The model receives the message and 77 possible labels, but not the correct label.
started = time.perf_counter()
result = model.classify_text(text, {"banking_intent": labels})
predict_seconds = time.perf_counter() - started

gold = example["output"]["classifications"][0]["true_label"][0]
print(f"Message: {text}")
print(f"Prediction: {result['banking_intent']}")
print(f"Correct label: {gold}")
print(f"Match: {result['banking_intent'] == gold}")
print(f"Load: {load_seconds:.2f}s; first prediction: {predict_seconds:.2f}s")
