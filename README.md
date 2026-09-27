# Fine-tuning GLiNER 2.5 on BANKING77

A local intent-classification experiment inspired by [Josh Kuechly's BANKING77 post](https://x.com/JoshKuechly/status/2100709039613100112). It compares the untouched [`fastino/gliner2.5-base-v1`](https://huggingface.co/fastino/gliner2.5-base-v1) checkpoint with a LoRA adapter trained on BANKING77. This is also a rehearsal for later work on extracting memory facts from personal conversations and notes; that extraction task is not implemented here.

## Official test result

Both models classified the same **3,080 official test messages** among all 77 BANKING77 intents. The adapted model was trained on 9,000 messages from the official training split; 1,003 training messages were reserved for development validation.

| Model | Correct | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Untouched GLiNER 2.5 base | 2,101 / 3,080 | 68.21% | 0.675 |
| Base + BANKING77 LoRA adapter | 2,784 / 3,080 | 90.39% | 0.904 |

The adapter gained **22.18 percentage points** of test accuracy.  Training ran for three epochs on an RTX 3050 Ti laptop GPU with 4 GB VRAM and took 110.7 minutes. The epoch-3 adapter was selected by development-validation loss before this test run.

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
- **Outputs:** Generated data, checkpoints, and paired predictions are written under `data/` and `runs/`, which are ignored by Git. The selected adapter and its label manifest are bundled in `artifacts/banking77-lora/` for the quick example above. Paired validation and test predictions are saved to `runs/development_validation.json` and `runs/development_test.json`.

## Reproduce the development run

Use the virtual environment installed above. The quick example works on CPU; `train_development.py` requires CUDA.

On the Windows laptop used for the training result, PyTorch `2.6.0+cu124` was installed with:

```powershell
.\.venv\Scripts\python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Choose the PyTorch build appropriate for your own GPU and driver. Prepare the pinned dataset, make the development split, train, and score both splits:

```powershell
.\.venv\Scripts\python banking77.py prepare
.\.venv\Scripts\python make_validation.py
.\.venv\Scripts\python train_development.py
.\.venv\Scripts\python score_validation.py
.\.venv\Scripts\python score_test.py
```

Training saves the adapter at `runs/development_lora/best/`. It refuses to overwrite an existing `runs/development_lora/` directory. Both scoring scripts print accuracy and macro-F1 and save every paired prediction. `score_test.py` refuses to overwrite an existing test result. Run it after fixing the model and settings using development validation.

`predict_one.py`, `score_ten.py`, and `train_pilot.py` provide smaller walkthrough checks; `inspect_architecture.py` prints the base model's encoder and classifier layout.

## Further work

Measure inference time with a controlled protocol and inspect test-set failure patterns. A later fit on all 10,003 original training messages would be a separate experiment with settings fixed in advance. The personal-memory task will need its own extraction schema and annotated evaluation data.
