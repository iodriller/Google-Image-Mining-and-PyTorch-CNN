# zero-label

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)

Type what you want to classify. It handles the rest.

You give it a few search terms — `"golden retriever"`, `"siberian husky"`, whatever — and it goes off, grabs images from DuckDuckGo, fine-tunes a pretrained EfficientNet-B0 on them, and gives you back a working classifier. No dataset to curate, no images to label, no Selenium setup to fight with. Just a model that runs.

---

## Get started

```bash
pip install -e .
```

```bash
# Train a dog-breed classifier
# Grabs ~100 images per class, should take ~5 min on CPU or ~90 sec on GPU
python cli.py train "golden retriever" "siberian husky" "german shepherd"

# See what it thinks of a photo
python cli.py predict photo.jpg

# Or open the browser UI and drag images in → http://localhost:7860
python cli.py demo
```

That's genuinely it for the basic case.

---

## What's happening under the hood

Images land in `images/<query_name>/`, split 85/15 for training and validation. The backbone (EfficientNet-B0 pretrained on ImageNet) stays frozen — those weights are already good at recognising edges, textures, shapes. Only the final classifier head gets trained, which is why it converges fast even on small datasets.

Best checkpoint gets saved to `best_model.pt` whenever validation accuracy improves. Standard modern training setup: AdamW, cosine LR schedule, CrossEntropyLoss. If you want to fine-tune the whole network instead of just the head, pass `--no-freeze` — though you'll want 200+ images per class for that to help.

EfficientNet-B0 is a deliberate default. It's compact enough to train on CPU in a few minutes and accurate enough to produce classifiers that actually work with 50–150 images per class. Swap to `--backbone resnet18` if you want something even lighter.

---

## CLI usage

Rather than a table of every flag, here's a tour of what you'll actually use:

```bash
# The main workflow
python cli.py train "cat" "dog"
python cli.py train "cat" "dog" --n-images 150 --epochs 15
python cli.py train "cat" "dog" --backbone resnet18
python cli.py train "cat" "dog" --no-freeze        # fine-tune everything

# Already have images? Skip the download step
python cli.py train "cat" "dog" --skip-scrape

# Classify a single image — shows confidence bars in the terminal
python cli.py predict photo.jpg
python cli.py predict photo.jpg --top-k 5

# Download images without training (useful for inspecting what you're working with)
python cli.py scrape "tabby cat" "siamese cat" --n-images 80

# Open the Gradio demo
python cli.py demo
python cli.py demo --share    # generates a public link you can share

# Debugging slow or missing downloads
python cli.py --verbose train "cat" "dog"
```

Any command also accepts `--help` for the full options list.

---

## Python API

If you want to run this from a notebook or script, everything's importable:

```python
from autovision import run

model, classes = run(
    queries=["sports car", "pickup truck", "minivan"],
    n_images=150,
    epochs=10,
)
```

Or piece it together manually if you need more control:

```python
from autovision.scraper import ImageScraper
from autovision.config import Blueprint
from autovision.trainer import train, predict

ImageScraper().search_and_download("cat", n_images=100)
ImageScraper().search_and_download("dog", n_images=100)

model, classes = train(Blueprint(epochs=5, backbone="resnet18"))
results = predict("photo.jpg", top_k=3)
```

---

## What's inside

```
autovision/
├── config.py      Blueprint — one dataclass that holds all training settings
├── scraper.py     DuckDuckGo image downloader, no browser required
├── model.py       Pretrained backbone factory (EfficientNet-B0 or ResNet-18)
├── trainer.py     Training loop, checkpointing, inference
└── pipeline.py    One call: scrape + train
cli.py             Command-line interface
app.py             Gradio web demo
notebooks/
├── quickstart.ipynb           Fastest path to a working classifier
└── custom_classifier.ipynb    Step-by-step with explanations
```

---

## License

MIT © 2024
