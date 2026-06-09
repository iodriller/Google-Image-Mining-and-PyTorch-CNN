"""
AutoVision — Gradio web demo.

Launch
------
    python app.py
    python app.py --model-path my_model.pt --share
    python cli.py demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch

from autovision.config import Blueprint
from autovision.model import build_model
from autovision.trainer import DEVICE, image_pipeline


class _Classifier:
    """Loads a checkpoint once at startup and reuses the model for every request."""

    def __init__(self) -> None:
        self.model = None
        self.class_names: list[str] = []
        self.cfg: Blueprint | None = None

    def load(self, model_path: str) -> bool:
        if not Path(model_path).exists():
            return False
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        self.class_names = ckpt["classes"]
        self.cfg = ckpt["config"]
        self.model = build_model(
            len(self.class_names), self.cfg.backbone, freeze=False
        ).to(DEVICE)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return True

    def classify(self, image) -> dict[str, float]:
        if self.model is None or image is None:
            return {}
        transform = image_pipeline(self.cfg.img_size, augment=False)
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        return {
            cls.replace("_", " ").title(): float(p)
            for cls, p in zip(self.class_names, probs.tolist())
        }


def build_demo(model_path: str = "best_model.pt") -> gr.Blocks:
    clf = _Classifier()
    loaded = clf.load(model_path)

    if loaded:
        class_list = "  ·  ".join(c.replace("_", " ").title() for c in clf.class_names)
        status_md = f"**Model ready** — {len(clf.class_names)} classes: {class_list}"
    else:
        status_md = (
            "⚠️ **No model found.**  "
            "Run `python cli.py train <query1> <query2> ...` first, then relaunch the demo."
        )

    n_classes = len(clf.class_names) if loaded else 5

    with gr.Blocks(title="AutoVision", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# AutoVision\n"
            "### Zero-dataset image classifier — trained entirely from web searches\n"
            "Upload any image and the model will rank it against its trained categories."
        )
        gr.Markdown(status_md)

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Upload an image")
                classify_btn = gr.Button("Classify", variant="primary", size="lg")

            with gr.Column(scale=1):
                label_out = gr.Label(num_top_classes=n_classes, label="Predictions")

        # Classify on button click or as soon as an image is dropped in
        classify_btn.click(fn=clf.classify, inputs=img_input, outputs=label_out)
        img_input.change(fn=clf.classify, inputs=img_input, outputs=label_out)

        if loaded:
            gr.Markdown(
                f"**Backbone**: `{clf.cfg.backbone}`  ·  "
                f"**Input size**: {clf.cfg.img_size}×{clf.cfg.img_size}  ·  "
                f"**Classes**: `{', '.join(clf.class_names)}`"
            )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoVision Gradio demo")
    parser.add_argument("--model-path", default="best_model.pt")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    args = parser.parse_args()

    build_demo(args.model_path).launch(server_port=args.port, share=args.share)
