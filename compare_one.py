"""Compare the base model and trained adapter on one BANKING77 message."""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Checkpoint uses legacy list-valued extra_special_tokens metadata.*")
warnings.filterwarnings("ignore", message="Encoder rejected attn_implementation='sdpa'; falling back to 'eager'.*")

import torch
from gliner2 import AutoExtractor
from peft import PeftModel

if len(sys.argv) < 2:
    raise SystemExit('Usage: python compare_one.py "your banking message"')

root = Path(__file__).parent
adapter_path = root / "artifacts" / "banking77-lora"
labels = json.loads((adapter_path / "manifest.json").read_text(encoding="utf-8"))["labels"]
device = "cuda" if torch.cuda.is_available() else "cpu"

base = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1", map_location=device)
base.eval()
text = " ".join(sys.argv[1:])
with torch.inference_mode():
    base_result = base.classify_text(text, {"banking_intent": labels}, include_confidence=True)["banking_intent"]

trained = PeftModel.from_pretrained(base, str(adapter_path)).to(device)
trained.eval()
with torch.inference_mode():
    trained_result = trained.classify_text(text, {"banking_intent": labels}, include_confidence=True)["banking_intent"]

print(f"Base: {base_result['label']} ({base_result['confidence']:.1%})")
print(f"Trained: {trained_result['label']} ({trained_result['confidence']:.1%})")
