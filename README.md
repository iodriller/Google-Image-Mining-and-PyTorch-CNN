# AutoVision

> **Train a custom image classifier in two commands — no dataset required.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)

AutoVision turns plain search terms into a trained image classifier. Type what you want to classify, it scrapes images from the web, fine-tunes a pretrained EfficientNet-B0 on them, and hands you a model ready to use — from the command line or a Gradio web demo.

No dataset curation. No labelling. No API keys.

---

## Quick start

```bash
# Install
pip install -e .

# Train a dog-breed classifier
# (~100 images per class, ~5 min on CPU or ~90 sec on GPU)
python cli.py train "golden retriever" "siberian husky" "german shepherd"

# Classify any image
python cli.py predict photo.jpg

# Open the interactive web demo → http://localhost:7860
python cli.py demo
```

---

## How it works

```
Your search terms
       │
       ▼
DuckDuckGo image search        no browser · no API key · no Selenium
       │  ~100–200 images per class
       ▼
EfficientNet-B0                pretrained on ImageNet (torchvision)
       │  backbone frozen · classifier head fine-tuned
       ▼
best_model.pt                  saved whenever val accuracy improves
       │
       ▼
Gradio demo  /  CLI predict  /  Python API
```

### Under the hood

| Setting | Value |
|---------|-------|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Classifier head | `nn.Linear(1280, num_classes)` |
| Loss | `CrossEntropyLoss` |
| Optimizer | AdamW |
| LR schedule | Cosine annealing |
| Input size | 224 × 224 |
| Augmentation | Random crop · horizontal flip · color jitter |
| Device | Auto-detected: CUDA → MPS (Apple Silicon) → CPU |

EfficientNet-B0 is a deliberate choice here — small enough to train on CPU in minutes, accurate enough to produce real classifiers from 50–150 images per class. Swap to `--backbone resnet18` if you want something even lighter.

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
| `scrape QUERY...` | Download images only — no training yet |
| `demo` | Open the Gradio web UI |

### `train`

```bash
python cli.py train "golden retriever" husky --n-images 150 --epochs 15
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n-images` | `100` | Images downloaded per category |
| `--epochs` | `10` | Training epochs |
| `--backbone` | `efficientnet_b0` | `efficientnet_b0` or `resnet18` |
| `--freeze / --no-freeze` | freeze | Freeze backbone (transfer learning) |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--skip-scrape` | off | Train on images already in `images/` |
| `--model-path` | `best_model.pt` | Where to save the checkpoint |

### `predict`

```bash
python cli.py predict photo.jpg --top-k 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `best_model.pt` | Checkpoint to load |
| `--top-k` | `3` | How many top predictions to show |

### `demo`

```bash
python cli.py demo --share   # generates a public link
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `best_model.pt` | Checkpoint to load |
| `--port` | `7860` | Local port |
| `--share` | off | Generate a public Gradio link |

---

## Python API

Use it directly in a script or notebook — useful if you want to loop over
experiments or integrate this into a larger pipeline.

```python
from autovision import run

model, classes = run(
    queries=["sports car", "pickup truck", "minivan"],
    n_images=150,
    epochs=10,
)
```

Or step by step, if you want more control:

```python
from autovision.scraper import ImageScraper
from autovision.config import Blueprint
from autovision.trainer import train, predict

scraper = ImageScraper(images_dir="images")
scraper.search_and_download("cat", n_images=100)
scraper.search_and_download("dog", n_images=100)

model, classes = train(Blueprint(epochs=5, backbone="resnet18"))

results = predict("photo.jpg", top_k=3)
```

---

## Code map

```
autovision/
├── __init__.py     Public API: run(), train(), predict(), Blueprint
├── config.py       Blueprint — all hyperparameters in one dataclass
├── scraper.py      DuckDuckGo image downloader (no Selenium)
├── model.py        Pretrained backbone factory (EfficientNet-B0, ResNet-18)
├── trainer.py      Training loop, data loading, checkpointing, inference
└── pipeline.py     One-call entry point: scrape → train
cli.py              Typer CLI (train / predict / scrape / demo)
app.py              Gradio web demo
notebooks/
├── quickstart.ipynb          Fastest path to a working classifier
└── custom_classifier.ipynb   Step-by-step with full control
```

---

## License

MIT © 2024
