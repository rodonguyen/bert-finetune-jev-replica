# GLiNER 2.5 → BANKING77 → personal memory graph

This is the code workspace for the project note in `D:/RodoVault/Projects/GLiNER-2.5-Banking77-to-Memory-Graph.md`. We are building it one step at a time. The first step prepares the [original BANKING77 data](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) for the [GLiNER2 classification training format](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md).

## Step 1: understand the data

The data source is pinned to Git commit `57ec275d8078af65b7731c2a98be812d844a6d6b`. BANKING77 has 77 banking intents, 10,003 training messages, and 3,080 test messages. Its license is CC BY 4.0. The generated `data/` folder is ignored by git.

```powershell
python banking77.py prepare
python banking77.py inspect
```

`prepare` downloads the two official CSV splits and category list. It writes `data/train.jsonl` and `data/test.jsonl` in GLiNER's classification format, plus `data/manifest.json` with the source commit, counts, labels, and source-file hashes. `inspect` prints three examples so you can see what the model will learn.

Example structure:

```json
{"input":"I am still waiting on my card?","output":{"classifications":[{"task":"banking_intent","labels":["card_arrival","card_linking"],"true_label":["card_arrival"]}]}}
```

The example above shortens the `labels` list for readability; the prepared files contain all 77 choices in every record.

## Step 2: one untouched-model prediction

Install the pinned dependencies and run one prediction using the untouched [GLiNER 2.5 base checkpoint](https://huggingface.co/fastino/gliner2.5-base-v1):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python predict_one.py
```

`predict_one.py` takes the first **training** message and supplies all 77 possible labels to the model. It prints the model prediction, then reveals the correct label for inspection. The model does not receive the correct label. First use downloads the weights and includes warm-up time, so this single timing is not a benchmark.

Josh Kuechly's screenshot does not identify his exact checkpoint or training settings. We chose the official English base checkpoint and will record that choice in any later comparison. Next we can write a fixed-test evaluator and then the fine-tuning step. BANKING77 classification is the rehearsal; the later 6190 task is memory-graph extraction from personal conversations and notes. No model training or test-set evaluation has been run yet.
