# CLAUDE.md

## Identity and attribution

All work done in this repo must appear to come from the repo owner, not from any AI tool.

- **Git identity**: always commit as `oney erge <oneyerge@gmail.com>`. Run `git config user.name "oney erge" && git config user.email "oneyerge@gmail.com"` at the start of any session before touching git.
- **Commit messages**: never append session URLs, tool attribution, or any reference to Claude, Anthropic, or AI assistants. Write the message as a developer would — what changed and why.
- **Branch names**: never use "claude", "anthropic", "ai", or any AI tool name as a prefix or component of a branch name. Use conventional prefixes: `feature/`, `fix/`, `refactor/`, `docs/`, `chore/`.
- **Code comments, docstrings, PR titles, PR bodies**: no mentions of Claude, Anthropic, or AI-generated content anywhere.
- **Contributor list**: the only contributor that should ever appear is `iodriller`. Verify with `git log --format="%an <%ae>"` before pushing.

## Project overview

`zero-label` — train a custom image classifier from plain search terms with no pre-built dataset. Type what you want to classify; it scrapes, trains, and serves a model.

Stack: Python 3.10+, PyTorch 2.x, torchvision, DuckDuckGo search, httpx, Gradio, typer.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests (no network required)
pytest tests/ -v

# Single test file
pytest tests/test_model.py -v

# Lint / format
ruff check autovision/ cli.py app.py
ruff format autovision/ cli.py app.py

# Type check
mypy autovision/ --ignore-missing-imports

# Train
autovision train --queries "golden retriever" "labrador" --n-images 80

# Predict
autovision predict image.jpg

# Gradio demo
autovision demo
```

## Code style

- Type hints on all public functions.
- No comments that describe *what* the code does — only the *why* when it's non-obvious.
- No docstring novels; one short line max if anything.
- `Blueprint` is the single source of truth for all training hyperparameters — add new ones there, not as function arguments.
- `_NUM_WORKERS = 0 if os.name == "nt" else 2` pattern must be kept; removing it breaks Windows.
- Validate only at system boundaries (user input, external URLs). Trust internal code.

## Git workflow

- Commit message format: `type: short description` — types are `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
- One logical change per commit. Don't bundle unrelated files.
- Run `pytest tests/ -v` before every commit.
- Never force-push `master`.
- Never skip hooks (`--no-verify`).

## Gotchas

- `torch.load(..., weights_only=False)` is intentional — the checkpoint stores a `Blueprint` dataclass which requires full pickle.
- Two separate `ImageFolder` instances in `_partition_data` (same indices, different transforms) — do not collapse into one; that would apply augmentation to validation data.
- `num_workers=0` on Windows is not a bug; it prevents CUDA + multiprocessing deadlocks.
- Tests use a synthetic in-memory fixture (`conftest.py`); no real images or network calls needed.
- DuckDuckGo rate-limits aggressive scrapers; `timeout=10` and small batches are intentional.

## What not to do

- Don't add features beyond what was asked.
- Don't refactor surrounding code when fixing a targeted bug.
- Don't add error handling for things that can't happen inside the package boundary.
- Don't create `*.md` documentation files unless explicitly asked.
- Don't design for hypothetical future requirements.
