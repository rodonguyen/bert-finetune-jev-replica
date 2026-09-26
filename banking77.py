"""Download and inspect the original BANKING77 data for a GLiNER 2.5 experiment."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{COMMIT}/banking_data/"


def download(name):
    with urlopen(SOURCE + name, timeout=30) as response:
        return response.read()


def prepare(folder):
    """Convert the pinned official CSV split to GLiNER classification JSONL."""
    labels = json.loads(download("categories.json"))
    source_files = {split: download(f"{split}.csv") for split in ("train", "test")}
    rows = {
        split: list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        for split, content in source_files.items()
    }
    if len(labels) != 77 or len(rows["train"]) != 10003 or len(rows["test"]) != 3080:
        raise ValueError("Unexpected BANKING77 size; inspect the source before using it")

    folder.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as output:
            for row in rows[split]:
                if row["category"] not in labels:
                    raise ValueError(f"Unknown category: {row['category']}")
                example = {
                    "input": row["text"],
                    "output": {"classifications": [{
                        "task": "banking_intent",
                        "labels": labels,
                        "true_label": [row["category"]],
                    }]},
                }
                output.write(json.dumps(example, ensure_ascii=False) + "\n")

    manifest = {
        "source": SOURCE,
        "commit": COMMIT,
        "counts": {split: len(rows[split]) for split in rows},
        "sha256": {split: hashlib.sha256(source_files[split]).hexdigest() for split in rows},
        "labels": labels,
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Prepared 10,003 train and 3,080 test examples across 77 intents.")


def inspect(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    print(f"Source commit: {manifest['commit']}")
    print(f"Rows: {manifest['counts']}; labels: {len(manifest['labels'])}")
    shown = set()
    with (folder / "train.jsonl").open(encoding="utf-8") as source:
        for line in source:
            example = json.loads(line)
            gold = example["output"]["classifications"][0]["true_label"][0]
            if gold in shown:
                continue
            print(f"{gold}: {example['input']}")
            shown.add(gold)
            if len(shown) == 3:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "inspect"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    {"prepare": prepare, "inspect": inspect}[args.command](args.data)
