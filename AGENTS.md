# AGENTS.md

## Project

Zero Label (`zero-label`, package code under `autovision/`) builds an image
classifier from search terms: it downloads images, trains a TorchVision backbone,
saves a checkpoint, and exposes CLI, Python, and Gradio interfaces.

Core boundaries:

- `autovision/scraper.py` owns external image discovery and downloads.
- `autovision/config.py` owns training configuration.
- `autovision/model.py` owns supported backbone construction.
- `autovision/trainer.py` owns training, checkpointing, and inference.
- `autovision/pipeline.py` composes the end-to-end workflow.
- `cli.py` and `app.py` are thin user interfaces over those modules.

## Commands

Use the committed uv lockfile for development:

```bash
uv sync --frozen --extra dev
uv run --frozen ruff check .
uv run --frozen pytest -q
uv run python cli.py --help
uv run python cli.py demo
```

User-facing examples also support `pip install -e .` and the installed
`autovision` console command.

## Project Rules

- Preserve the separation between scraping, configuration, model construction,
  training, and interfaces.
- Keep network, pretrained-weight, filesystem, and device selection behavior
  explicit. Tests should not silently depend on live search results or downloads.
- Keep class ordering, checkpoint metadata, transforms, and inference labels
  consistent when changing training behavior.
- Do not commit downloaded datasets, checkpoints, generated arrays, caches, or
  notebook output.
- Treat a public Gradio share link as an external exposure and require explicit
  user intent before starting one.
- Avoid expanding supported backbones or training options without a concrete use
  case and focused coverage.

## Verification

- Documentation or guidance only: verify referenced paths and run
  `git diff --check`; application tests are not required.
- Python behavior: run the focused test, then `uv run --frozen pytest -q`.
- Python quality: run `uv run --frozen ruff check .`.
- CLI changes: exercise the affected command with `--help` or a bounded local
  fixture; do not trigger search, model downloads, or training unnecessarily.

Report network-dependent, GPU-dependent, or large-model checks as skipped unless
they were actually observed.

## Git and Safety

- Preserve unrelated changes and keep commits focused.
- Use the configured repository-owner identity.
- Do not add assistant names, co-author trailers, session links, or tool
  attribution to Git artifacts.
- Review the source and license of downloaded images before redistributing data or
  generated models.
