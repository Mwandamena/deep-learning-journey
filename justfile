# Run `just` with no args to see this list
default:
    @just --list

# --- digit-classifier ---

train-digits:
    cd apps/digit-classifier && uv run python -m train.train

serve-digits:
    cd apps/digit-classifier && uv run python main.py

# --- fashion-mnist ---

train-fashion:
    cd apps/fashion-mnist && uv run python -m train.train

serve-fashion:
    cd apps/fashion-mnist && uv run python main.py

dev-digits:
    cd apps/digit-classifier && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-fashion:
    cd apps/fashion-mnist && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8001

# --- workspace-wide ---

sync:
    uv sync

clean-checkpoints app:
    rm -f apps/{{app}}/checkpoints/*.pt
    @echo "Cleared checkpoints for {{app}}"