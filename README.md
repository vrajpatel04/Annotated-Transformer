# Annotated Transformer — Modular Implementation

A modular, object-oriented port of [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) notebook, trained on the **Multi30k** German–English machine translation dataset.

## Features

- **Modular OOP architecture** — model, data, training, evaluation, and dashboard are separate packages
- **Configurable training** — epochs, batch size, learning rate, warmup, model size, and more via YAML or CLI
- **Evaluation pipeline** — `corpus_perplexity` and `corpus_bleu` on the validation set each epoch
- **Interactive dashboard** — Plotly HTML report with loss, accuracy, perplexity, KL divergence, BLEU, and GPU temperature

## Project Structure

```
src/transformer/
├── config.py              # TrainingConfig dataclass
├── cli.py                 # CLI entry point
├── model/                 # Transformer architecture
├── data/                  # Multi30k loading & tokenization
├── training/              # Trainer, loss, scheduler
├── evaluation/            # corpus_bleu & corpus_perplexity
└── dashboard/             # Metrics tracking & Plotly charts
configs/default.yaml       # Default hyperparameters
main.py                    # Backward-compatible wrapper
```

## Setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Python 3.13 is pinned in `.python-version`.

```bash
# Install all dependencies into .venv (includes spaCy language models)
uv sync
```

The Multi30k dataset is loaded from Hugging Face: [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k).

### GPU support

By default, `uv sync` installs a **CUDA-enabled PyTorch** build (`cu124`). Training uses `device: auto` from `configs/default.yaml`, which picks `cuda` when a compatible NVIDIA GPU is available and otherwise falls back to CPU.

```bash
# CUDA PyTorch (GPU)
uv sync
```

For a CPU-only PyTorch install (smaller download, no NVIDIA GPU required), reinstall torch from the CPU index after syncing:

```bash
uv sync
uv pip install --reinstall torch --index https://download.pytorch.org/whl/cpu
```

Verify GPU detection:

```bash
uv run python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Force a device at runtime:

```bash
uv run train-transformer --device cuda
uv run train-transformer --device cpu
uv run train-transformer --device cuda:0
```

If upgrading from an older setup, close any terminals using `.venv`, then rerun `uv sync` so the environment is recreated on Python 3.13. Delete `outputs/vocab.pt` if present so vocabularies are rebuilt without torchtext.

## Training

```bash
# Train with default config
uv run train-transformer --mode train

# Override hyperparameters
uv run train-transformer --epochs 10 --batch-size 64 --lr 1.0 --output-dir outputs

# Use a custom config file
uv run train-transformer --config configs/default.yaml
```

Outputs are written to `outputs/`:
- `multi30k_model_XX.pt` — per-epoch checkpoints
- `multi30k_model_final.pt` — final model weights
- `vocab.pt` — cached source/target vocabularies
- `metrics.json` — per-epoch metrics history
- `dashboard.html` — interactive training dashboard

## Evaluation

```bash
uv run train-transformer --mode evaluate --checkpoint outputs/multi30k_model_final.pt
```

## Configuration

Edit `configs/default.yaml` or pass CLI flags:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 8 | Training epochs |
| `batch_size` | 32 | Batch size |
| `accum_iter` | 10 | Gradient accumulation steps |
| `base_lr` | 1.0 | Adam base learning rate |
| `warmup` | 3000 | LR warmup steps |
| `d_model` | 512 | Model dimension |
| `num_layers` | 6 | Encoder/decoder layers |
| `label_smoothing` | 0.1 | KL label smoothing factor |
| `device` | auto | `auto`, `cpu`, `cuda`, or `cuda:0` |

## Dashboard Metrics

The generated `dashboard.html` includes:

- **Loss** — train vs validation (per token)
- **Accuracy** — token-level prediction accuracy
- **Perplexity** — exp(loss), including corpus-level
- **KL Divergence** — label smoothing KL loss
- **Corpus BLEU** — sacrebleu corpus score with greedy decoding
- **GPU Temperature** — recorded each epoch (when GPU is available)

## Python API

Run scripts inside the uv environment:

```python
# uv run python
from transformer import TrainingConfig, Trainer

config = TrainingConfig(num_epochs=5, batch_size=16)
trainer = Trainer(config)
metrics = trainer.train()

results = trainer.evaluate_only("outputs/multi30k_model_final.pt")
print(results["corpus_bleu"], results["corpus_perplexity"])
```

## uv Commands Reference

| Command | Description |
|---------|-------------|
| `uv sync` | Install/sync dependencies (including spaCy models) |
| `uv add <pkg>` | Add a new dependency |
| `uv run train-transformer` | Run the training CLI |

## Reference

Based on the Harvard NLP [Annotated Transformer notebook](Notebooks/AnnotatedTransformer.ipynb) implementing [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
