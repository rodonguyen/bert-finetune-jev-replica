"""Score 10 fixed training examples with all 77 BANKING77 labels."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from gliner2 import AutoExtractor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
args = parser.parse_args()
if args.device == "cuda" and not torch.cuda.is_available():
    parser.error("CUDA is unavailable in this PyTorch environment")

DATA = Path(__file__).parent / "data"
labels = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["labels"]
with (DATA / "train.jsonl").open(encoding="utf-8") as source:
    train_rows = [json.loads(line) for line in source]
examples = random.Random(42).sample(train_rows, 10)

model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1", map_location=args.device)
model.eval()
correct = 0
if args.device == "cuda":
    torch.cuda.synchronize()
started = time.perf_counter()
with torch.inference_mode():
    for example in examples:
        prediction = model.classify_text(
            example["input"], {"banking_intent": labels}
        )["banking_intent"]
        gold = example["output"]["classifications"][0]["true_label"][0]
        correct += prediction == gold
        print(f"{prediction == gold} | predicted={prediction} | gold={gold}")
        if prediction != gold:
            print(f"  text: {example['input']}")

if args.device == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - started
print(f"Device: {torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
print(f"Accuracy: {correct}/10 = {correct / 10:.1%}")
print(f"Prediction time for 10 messages: {elapsed:.2f}s")
