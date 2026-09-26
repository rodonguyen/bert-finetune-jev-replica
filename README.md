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

`predict_one.py` takes the first **training** message and supplies the first five labels in the manifest, as selected for this small demonstration. It prints the model prediction and confidence score, then reveals the correct label for inspection. The model does not receive the correct label. The score is not a calibrated probability of correctness, even when formatted as a percentage. The later `score_ten.py` and `compare_pilot.py` runs use all 77 labels. First use downloads the weights and includes warm-up time, so this single timing is not a benchmark.

On the first local CPU walkthrough (2026-09-27), “I am still waiting on my card?” was predicted as `card_arrival`, matching the correct label. Loading took 47.78 s including the first download; the prediction took 1.23 s. One correct message does not estimate dataset accuracy.

## Step 3: check an evaluation loop

```powershell
.\.venv\Scripts\python score_ten.py --device cpu
```

`score_ten.py` picks 10 training messages with a fixed random seed, supplies all 77 labels for each, and counts matches. `model.eval()` and `torch.inference_mode()` make this a forward-pass check like evaluating a CNN. The sample is deliberately small and comes from the training split, so its accuracy is only a code and error-inspection check. The official test split remains untouched.

The first ten-message check scored 6/10 and took 11.07 s of prediction time on CPU, excluding model loading. Inspect the mismatched message text before drawing conclusions from that fraction.

### CUDA on this Windows laptop

The RTX 3050 Ti laptop GPU and driver 551.78 support the CUDA 12.4 wheel used here. The default PyPI installation provided a CPU-only PyTorch wheel, so install the official CUDA build **inside this project's `.venv`** after `requirements.txt`:

```powershell
.\.venv\Scripts\python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python score_ten.py --device cuda
```

The 2026-09-27 CUDA run used PyTorch `2.6.0+cu124` and scored the same 6/10 in 2.74 s of prediction time, excluding checkpoint loading. The earlier CPU run used PyTorch `2.14.0+cpu`, so those two times are observations under different software builds, not a controlled speed comparison. The official test set has not been evaluated.

## Step 4: make a validation split before training

```powershell
python make_validation.py
```

This makes a seeded 90/10 split of the **official training** messages. It groups duplicate text, keeps all 77 intents in both subsets, and checks that no identical message crosses between them. Use `data/development_train.jsonl` to try training settings and `data/validation.jsonl` to compare them. The official `data/test.jsonl` remains the final comparison set. If we choose settings on validation, we can later retrain on all 10,003 original training messages before opening the test split.

## Step 5: two training updates on CUDA

```powershell
.\.venv\Scripts\python train_pilot.py
```

This samples eight development-training messages and runs **two** LoRA updates on the same GLiNER 2.5 base checkpoint. Batch size is one for the 4 GB GPU. It prints the fraction of trainable parameters and saves a small adapter under `runs/pilot_lora/final/`. The script verifies that two update steps completed and the adapter file exists. This is a training-path check, not a tuned classifier or a test-set result. LoRA is an explicit pilot choice for the local GPU; Josh's post did not specify his fine-tuning method. [GLiNER2 training guide](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md)

The 2026-09-27 pilot completed two optimizer steps in 2.0 s after setup. It trained 1,327,104 of 194,908,695 parameters (0.68%) and saved a 5.1 MB adapter. The two minibatch losses were 0.1300 and 0.1501; they came from different examples and do not show whether the model improved. Full validation and official test accuracy remain unmeasured.

## Step 6: compare the saved adapter on validation data

```powershell
.\.venv\Scripts\python compare_pilot.py
```

This samples the same 20 held-out validation messages for both runs. It first predicts with the untouched checkpoint, then loads the saved LoRA adapter onto that checkpoint and predicts again with the same 77 labels. Paired outputs go to `runs/pilot_comparison.json`. This small check uses no official test examples and is not an accuracy benchmark.

The 2026-09-27 check got **15/20** for both models, with **0 changed predictions**. Two optimizer updates on eight training messages made no visible difference on this sample. The adapter still needs a meaningful training run and broader validation before judging whether fine-tuning helps.

Josh Kuechly's screenshot does not identify his exact checkpoint or training settings. We chose the official English base checkpoint and will record that choice in any later comparison. BANKING77 classification is the rehearsal; the later 6190 task is memory-graph extraction from personal conversations and notes. The official test set has not been evaluated.

## Step 7: train on the development split, then score validation

From this repository folder, run these commands in order:

```powershell
.\.venv\Scripts\python train_development.py
.\.venv\Scripts\python score_validation.py
```

The first command trains a fresh LoRA adapter for **three epochs** on all **9,000 development-training messages**. A micro-batch holds one message on the 4 GB GPU; four micro-batches accumulate before each optimizer update, giving an effective batch of four and **6,750 updates** across three epochs. It checks validation loss after each epoch and saves the lowest-loss adapter under `runs/development_lora/best/`; it does not resume the two-step pilot. The second command scores the untouched checkpoint and that selected adapter on the same **1,003 validation messages**, using all 77 labels. It prints accuracy, macro-F1, and the accuracy difference in percentage points, then saves paired predictions to `runs/development_validation.json`.

The development run uses LoRA rank 8, alpha 16, dropout 0.1, and targets the encoder plus the classification head. AdamW uses a peak LoRA learning rate of `5e-4`, weight decay `0.01`, gradient clipping at `1.0`, and a linear schedule with 10% warmup. FP16, a one-message evaluation batch, and the tested training micro-batch keep memory demand modest. These are starting settings, selected for this development run rather than claimed to match the X post.

A local attempt with physical batch size 20 failed on the first six batches with CUDA out-of-memory errors and saved no adapter. Its diagnostics are preserved under `runs/development_lora_failed_b20_20260927/`. Gradient accumulation changes the optimizer's effective batch without holding 20 messages' activations in GPU memory at once.

Training may take substantially longer than the two-step pilot. Keep the laptop powered and allow the Python process to finish before running the validation command. Both scripts require CUDA. The validation data selects the best epoch but supplies no gradient updates. The official test split is untouched; after inspecting validation results, we can retrain the selected setup on all 10,003 original training messages and evaluate on the official test set. Re-running training stops if `runs/development_lora/` already exists, protecting the first run from accidental overwrite.

## Inspect the base checkpoint

```powershell
.\.venv\Scripts\python inspect_architecture.py
```

This prints the loaded encoder configuration, top-level parameter counts, and BANKING77 classification head. The base checkpoint has 193,581,591 parameters: 183,763,200 in the 12-layer DeBERTa encoder, 1,182,721 in the classifier, and the remainder in extraction, record, and relation heads. The classifier receives contextual label-marker states from the encoder and scores each supplied intent label. BANKING77 fine-tuning targets the encoder and classifier; the later memory-graph task will need the extraction and relation paths as well.
