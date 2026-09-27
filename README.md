# Fine-tuning GLiNER 2.5 on BANKING77

A local intent-classification experiment inspired by [Josh Kuechly's BANKING77 post](https://x.com/JoshKuechly/status/2100709039613100112). It compares the untouched [`fastino/gliner2.5-base-v1`](https://huggingface.co/fastino/gliner2.5-base-v1) checkpoint with a LoRA adapter trained on BANKING77. This is also a rehearsal for later work on extracting memory facts from personal conversations and notes; that extraction task is not implemented here.

## Development results

Both models classified the same **1,003 held-out validation messages** among all 77 BANKING77 intents.

| Model | Correct | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Untouched GLiNER 2.5 base | 657 / 1,003 | 65.50% | 0.650 |
| Base + BANKING77 LoRA adapter | 900 / 1,003 | 89.73% | 0.899 |

The adapter gained **24.23 percentage points** of accuracy. It fixed 256 base-model errors and introduced 13 new ones. The selected adapter was trained for three epochs on the other **9,000 training messages**; training took 110.7 minutes on an RTX 3050 Ti laptop GPU with 4 GB VRAM. The best checkpoint was selected using validation loss.

**These are development results, not an official test-set score.** The same validation split selected the checkpoint and supplied the reported accuracy. The 3,080-row official test split has not been evaluated. This project has not run Jev, and the post does not disclose enough settings for an exact reproduction of its numbers.

## Try it without training

The [5.20 MiB adapter](artifacts/banking77-lora/adapter_model.safetensors), its configuration, and the 77-label list are included in this repository. After cloning, install dependencies and compare both models on one message:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python compare_one.py "I don't have my card in 1 week. Should I be worried?"
```

For this held-out validation message, the saved paired predictions were:

| Input | Untouched base | Trained adapter |
| --- | --- | --- |
| “I don't have my card in 1 week. Should I be worried?” | `card_about_to_expire` | `card_arrival` |

The command prints **two lines**, one label and confidence percentage per model. The percentages are model scores, not calibrated probabilities. `compare_one.py` runs on CPU if CUDA is unavailable. First use downloads the base checkpoint (about 739 MiB) from Hugging Face; the adapter alone is not a complete model. No BANKING77 dataset download or training is needed for this example.

## Experiment

- **Data:** [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), pinned to source commit `57ec275d8078af65b7731c2a98be812d844a6d6b` (10,003 train, 3,080 test, 77 intents; CC BY 4.0). `banking77.py` records source hashes in `data/manifest.json`.
- **Development split:** `make_validation.py` divides the official training rows into 9,000 training and 1,003 validation examples with seed 42. Duplicate normalized messages stay in one subset, and every intent appears in both.
- **Model:** `fastino/gliner2.5-base-v1` with `gliner2==2.0.0`. The untouched and adapted models receive the same text and all 77 candidate labels.
- **Training:** LoRA on the encoder and classification head; rank 8, alpha 16, dropout 0.1; physical batch 1 with four gradient-accumulation steps (effective batch 4); three epochs; AdamW learning rate `5e-4`, 10% linear warmup, weight decay `0.01`, gradient clipping `1.0`, FP16. The lowest validation-loss checkpoint is used for scoring.
- **Outputs:** Generated data, checkpoints, and paired predictions are written under `data/` and `runs/`, which are ignored by Git. The selected development adapter and its label manifest are bundled in `artifacts/banking77-lora/` for the quick example above. The development score file is `runs/development_validation.json` after running the scripts.

## Reproduce the development run

Use the virtual environment installed above. The quick example works on CPU; `train_development.py` requires CUDA.

On the Windows laptop used for the training result, PyTorch `2.6.0+cu124` was installed with:

```powershell
.\.venv\Scripts\python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Choose the PyTorch build appropriate for your own GPU and driver. Prepare the pinned dataset, make the development split, train, and score:

```powershell
.\.venv\Scripts\python banking77.py prepare
.\.venv\Scripts\python make_validation.py
.\.venv\Scripts\python train_development.py
.\.venv\Scripts\python score_validation.py
```

Training saves the adapter at `runs/development_lora/best/`. It refuses to overwrite an existing `runs/development_lora/` directory. Scoring prints accuracy and macro-F1 for both models and saves every paired prediction. The official `data/test.jsonl` is prepared but is not read by these training or scoring scripts.

`predict_one.py`, `score_ten.py`, and `train_pilot.py` provide smaller walkthrough checks; `inspect_architecture.py` prints the base model's encoder and classifier layout.

## Next step

Fit the selected training setup on all 10,003 original training messages, then evaluate the untouched checkpoint and final adapter once on the 3,080-row official test split. Report test accuracy, macro-F1, paired errors, and inference timing. The later personal-memory task will need its own extraction schema and annotated evaluation data.
