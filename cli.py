"""
autovision CLI — scrape, train, predict.

Examples
--------
    # Download images for 3 categories and train a classifier
    python cli.py train "golden retriever" "siberian husky" "german shepherd"

    # Same but 200 images per class, 15 epochs, ResNet-18 backbone
    python cli.py train cat dog --n-images 200 --epochs 15 --backbone resnet18

    # Skip downloading and train on images you already have
    python cli.py train cat dog --skip-scrape

    # Download images only, no training
    python cli.py scrape "sports car" "pickup truck" --n-images 80

    # Predict on an image
    python cli.py predict photo.jpg
    python cli.py predict photo.jpg --model-path my_model.pt --top-k 5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.table import Table

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

console = Console()
app = typer.Typer(
    name="autovision",
    help="Zero-dataset image classifier — scrape images from the web, train, predict.",
    add_completion=False,
)


@app.callback()
def _setup(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug output (e.g. failed image downloads)."),
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@app.command()
def train(
    queries: List[str] = typer.Argument(..., help="One search term per category."),
    n_images: int = typer.Option(100, "--n-images", "-n", help="Images to download per category."),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Training epochs."),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    backbone: str = typer.Option("efficientnet_b0", "--backbone", help="efficientnet_b0 | resnet18"),
    freeze: bool = typer.Option(True, "--freeze/--no-freeze", help="Freeze backbone (transfer learning)."),
    images_dir: str = typer.Option("images", "--images-dir", help="Directory for downloaded images."),
    model_path: str = typer.Option("best_model.pt", "--model-path", help="Where to save the trained model."),
    skip_scrape: bool = typer.Option(False, "--skip-scrape", help="Skip download, train on existing images."),
):
    """Scrape images for each QUERY and train a custom image classifier."""
    from autovision.scraper import ImageScraper
    from autovision.config import Blueprint
    from autovision.trainer import train as run_train

    console.rule("[bold blue]AutoVision — Train[/bold blue]")
    console.print(f"Categories : {queries}")
    console.print(f"Backbone   : {backbone}  |  Freeze: {freeze}  |  Epochs: {epochs}")

    if not skip_scrape:
        scraper = ImageScraper(images_dir=images_dir)
        for q in queries:
            scraper.search_and_download(q, n_images=n_images)

    cfg = Blueprint(
        images_dir=images_dir,
        model_path=model_path,
        backbone=backbone,
        freeze_backbone=freeze,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    run_train(cfg)
    console.print(f"\n[bold green]Done![/bold green] Model saved → [cyan]{model_path}[/cyan]")


@app.command()
def predict(
    image: Path = typer.Argument(..., help="Path to an image file."),
    model_path: str = typer.Option("best_model.pt", "--model-path", help="Path to best_model.pt."),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of top predictions to display."),
):
    """Classify an image using a trained model and show confidence scores."""
    from autovision.trainer import predict as run_predict

    if not Path(model_path).exists():
        console.print(f"[red]Model not found:[/red] {model_path}")
        raise typer.Exit(1)

    if not image.exists():
        console.print(f"[red]Image not found:[/red] {image}")
        raise typer.Exit(1)

    results = run_predict(str(image), model_path=model_path, top_k=top_k)

    table = Table(title=f"Predictions  ·  {image.name}", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=5, justify="center")
    table.add_column("Class", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("", justify="left")

    for rank, (cls, prob) in enumerate(results, 1):
        bar = "█" * round(prob * 25)
        style = "green" if rank == 1 else ""
        table.add_row(
            str(rank),
            cls.replace("_", " ").title(),
            f"{prob:.1%}",
            f"[{style}]{bar}[/{style}]" if style else bar,
        )

    console.print()
    console.print(table)


@app.command()
def scrape(
    queries: List[str] = typer.Argument(..., help="Search terms to download images for."),
    n_images: int = typer.Option(100, "--n-images", "-n", help="Images per category."),
    images_dir: str = typer.Option("images", "--images-dir", help="Output directory."),
):
    """Download images from DuckDuckGo without training."""
    from autovision.scraper import ImageScraper

    console.rule("[bold blue]AutoVision — Scrape[/bold blue]")
    scraper = ImageScraper(images_dir=images_dir)
    for q in queries:
        scraper.search_and_download(q, n_images=n_images)
    console.print(f"\n[green]All images saved to [cyan]{images_dir}/[/cyan][/green]")


@app.command()
def demo(
    model_path: str = typer.Option("best_model.pt", "--model-path", help="Path to best_model.pt."),
    port: int = typer.Option(7860, "--port", help="Local port to serve the demo on."),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio share link."),
):
    """Launch the Gradio web demo in your browser."""
    from app import build_demo

    console.rule("[bold blue]AutoVision — Demo[/bold blue]")
    console.print(f"Model  : [cyan]{model_path}[/cyan]")
    console.print(f"URL    : [cyan]http://localhost:{port}[/cyan]")
    if share:
        console.print("[yellow]--share is on: a public link will be printed below.[/yellow]")

    build_demo(model_path).launch(server_port=port, share=share)


if __name__ == "__main__":
    app()
