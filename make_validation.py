"""Hold out 10% of BANKING77 training data for development checks."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data"
SEED = 42


def label_of(row):
    return row["output"]["classifications"][0]["true_label"][0]


def text_key(row):
    return " ".join(row["input"].casefold().split())


with (DATA / "train.jsonl").open(encoding="utf-8") as source:
    rows = [json.loads(line) for line in source]

# One normalized message stays in one subset, even if it occurs twice.
groups = defaultdict(list)
for index, row in enumerate(rows):
    groups[text_key(row)].append(index)

by_label = defaultdict(list)
for indices in groups.values():
    labels = {label_of(rows[index]) for index in indices}
    if len(labels) != 1:
        raise ValueError("The same message has conflicting labels")
    by_label[labels.pop()].append(indices)

rng = random.Random(SEED)
validation_indices = set()
for label in sorted(by_label):
    label_groups = by_label[label]
    rng.shuffle(label_groups)
    target = round(sum(len(group) for group in label_groups) * 0.10)
    selected = 0
    for group in label_groups:
        if selected >= target:
            break
        validation_indices.update(group)
        selected += len(group)

subsets = {
    "development_train": [row for i, row in enumerate(rows) if i not in validation_indices],
    "validation": [row for i, row in enumerate(rows) if i in validation_indices],
}
for name, subset in subsets.items():
    with (DATA / f"{name}.jsonl").open("w", encoding="utf-8") as output:
        for row in subset:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

train_texts = {text_key(row) for row in subsets["development_train"]}
validation_texts = {text_key(row) for row in subsets["validation"]}
train_labels = Counter(label_of(row) for row in subsets["development_train"])
validation_labels = Counter(label_of(row) for row in subsets["validation"])
if train_texts & validation_texts or len(train_labels) != 77 or len(validation_labels) != 77:
    raise ValueError("Split failed the overlap or class-coverage check")

summary = {
    "seed": SEED,
    "source": "data/train.jsonl",
    "counts": {name: len(subset) for name, subset in subsets.items()},
    "classes_in_each_subset": {"development_train": len(train_labels), "validation": len(validation_labels)},
    "shared_normalized_messages": len(train_texts & validation_texts),
}
(DATA / "split.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
