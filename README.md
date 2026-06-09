# AutoVision

> **Train a custom image classifier in two commands — no dataset required.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)

AutoVision turns plain search terms into a trained image classifier.  
Type what you want to classify → it scrapes images from the web → fine-tunes a pretrained EfficientNet-B0 → gives you a model you can run in a Gradio web demo or from the command line.

No dataset curation. No labelling. No API keys.

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Train a dog-breed classifier
#    (downloads ~100 images per class, fine-tunes EfficientNet-B0)
python cli.py train "golden retriever" "siberian husky" "german shepherd"

# 3. Classify any image
python cli.py predict photo.jpg

# 4. Launch the interactive web demo → http://localhost:7860
python cli.py demo
```

---

## How it works

```
Your search terms
       │
       ▼
DuckDuckGo image search          no browser · no API key · no Selenium
       │  ~100–200 images per class
       ▼
EfficientNet-B0                  pretrained on ImageNet (torchvision)
       │  backbone frozen · classifier head fine-tuned
       ▼
best_model.pt                    saved whenever val accuracy improves
       │
       ▼
Gradio demo  /  CLI predict  /  Python API
```

**Training details**

| Setting | Value |
|---------|-------|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Classifier head | `nn.Linear(1280, num_classes)` |
| Loss | `CrossEntropyLoss` |
| Optimizer | AdamW |
| LR schedule | Cosine annealing |
| Input size | 224 × 224 |
| Augmentation | Random crop · horizontal flip · color jitter |
| Device | Auto: CUDA → MPS (Apple Silicon) → CPU |

---

## Installation

**CPU (any platform):**
```bash
pip install -e .
```

**GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

Requires **Python 3.10+**.

---

## CLI reference

```
python cli.py COMMAND [OPTIONS]
```

| Command | What it does |
|---------|-------------|
| `train QUERY...` | Scrape images and train a classifier |
| `predict IMAGE` | Classify a single image, show confidence scores |
| `scrape QUERY...` | Download images only — skip training |
| `demo` | Open the Gradio web UI |

### `train` options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-images` | `100` | Images downloaded per category |
| `--epochs` | `10` | Training epochs |
| `--backbone` | `efficientnet_b0` | `efficientnet_b0` or `resnet18` |
| `--freeze / --no-freeze` | freeze | Freeze backbone for transfer learning |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--skip-scrape` | off | Skip download; train on existing `images/` folder |
| `--model-path` | `best_model.pt` | Where to save the trained model |

### `predict` options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `best_model.pt` | Checkpoint to load |
| `--top-k` | `3` | How many top predictions to show |

### `demo` options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `best_model.pt` | Checkpoint to load |
| `--port` | `7860` | Local port |
| `--share` | off | Generate a public Gradio share link |

---

## Python API

Use AutoVision directly in your own scripts or notebooks:

```python
from autovision import run

model, classes = run(
    queries=["sports car", "pickup truck", "minivan"],
    n_images=150,
    epochs=10,
)
```

Or step by step:

```python
from autovision.scraper import ImageScraper
from autovision.trainer import train
from autovision.config import TrainConfig

# Download images
scraper = ImageScraper(images_dir="images")
scraper.search_and_download("cat", n_images=100)
scraper.search_and_download("dog", n_images=100)

# Train
model, classes = train(TrainConfig(epochs=5, backbone="resnet18"))
```

---

## Project layout

```
autovision/
├── __init__.py     Public API: run(), train(), predict(), TrainConfig
├── config.py       TrainConfig dataclass — all hyperparameters in one place
├── scraper.py      DuckDuckGo image downloader
├── model.py        Pretrained backbone factory (EfficientNet-B0, ResNet-18)
├── trainer.py      Training loop, data loading, checkpointing, inference
└── pipeline.py     Orchestrator: scrape → train in one call
cli.py              Typer CLI  (train / predict / scrape / demo)
app.py              Gradio web demo
```

---

## License

MIT
