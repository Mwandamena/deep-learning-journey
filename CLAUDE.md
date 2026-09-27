# deep-learning-journey

A uv workspace monorepo for a structured deep learning curriculum. See `guide.md` for the full roadmap.

## Stack

- Python, PyTorch, FastAPI, uvicorn
- `uv` for workspace and dependency management
- `just` as a task runner (`justfile` at repo root)
- W&B (wandb) for training metrics
- `timm` for pretrained vision backbones

## Structure

```
apps/          individual ML apps (each is a uv workspace member)
packages/      shared libraries
  trainkit/    reusable Trainer class — reuse this in every new app
  api-middleware/  shared FastAPI error handlers + structured logging
guide.md       phase roadmap
justfile       task runner commands
```

## Commands

```bash
# list all tasks
just

# train / serve a specific app
just train-digits
just serve-digits
just train-fashion
just serve-fashion

# dev server with hot reload
just dev-digits     # port 8000
just dev-fashion    # port 8001

# sync workspace dependencies
just sync

# clear checkpoints
just clean-checkpoints <app>   # e.g. just clean-checkpoints digit-classifier
```

Run any Python directly with `uv run python ...` from the app's directory.

## Current phase

Phase 1 (CNNs, vision, transfer learning). `apps/birds-classifier` is the active Phase 1 project.

## Conventions

- **New apps** follow the pattern in `digit-classifier`: `main.py` (FastAPI app), `models/`, `routes/`, `train/`.
- **Training** always uses `packages/trainkit.Trainer`. Do not write one-off training loops.
- **Serving** always uses `packages/api-middleware` for error handlers and logging.
- **Checkpoints** go in `apps/<name>/checkpoints/` (already gitignored).
- Each app has its own `pyproject.toml` and is a `uv` workspace member.

## Tests

No automated test suite. This is a personal learning project; tests are not required. Do not add a test runner or flag missing coverage.

## Adding a new app

1. Create `apps/<name>/` with `pyproject.toml` declaring it as a package.
2. Add `"apps/<name>"` to `[tool.uv.workspace] members` in the root `pyproject.toml` if not using a glob.
3. Add `just` recipes for `train-<name>`, `serve-<name>`, `dev-<name>`.
4. Use `trainkit.Trainer` for training and `api_middleware` for the FastAPI server.

## Non-obvious decisions

- AMP (`torch.amp`) is enabled only when `device == "cuda"` — no-ops gracefully on CPU.
- `Trainer` saves `latest.pt` every epoch and `best.pt` only on improvement — resume with `resume_from="checkpoints/latest.pt"`.
- W&B is optional (`use_wandb=False` skips it) — useful when iterating locally without a run logged.
