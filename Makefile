# autoresearch — Modal deployment commands
# Usage: make <target>
# Run `make help` for a summary.

.PHONY: help install auth secret prepare check ping run results log stop

RUN_TAG ?= run
N       ?= 20

help:
	@echo ""
	@echo "autoresearch on Modal"
	@echo "─────────────────────────────────────────────────"
	@echo ""
	@echo "  FIRST TIME SETUP (run in this order):"
	@echo "    make install      Install local dev tools (modal, anthropic)"
	@echo "    make auth         Authenticate with Modal (opens browser)"
	@echo "    make secret       Set Anthropic API key in Modal"
	@echo "    make ping         Verify Modal connection (no GPU)"
	@echo "    make prepare      Download data + train tokenizer on GPU (one-time)"
	@echo "    make check        Verify data is ready"
	@echo ""
	@echo "  RUNNING EXPERIMENTS:"
	@echo "    make run          Launch autonomous loop (default 20 experiments)"
	@echo "    make run RUN_TAG=apr5 N=50   Named session, 50 experiments"
	@echo ""
	@echo "  INSPECTING RESULTS:"
	@echo "    make results      Print results.tsv for RUN_TAG"
	@echo "    make log          Tail last training run log"
	@echo ""
	@echo "  MANUAL SINGLE RUN (local agent mode):"
	@echo "    make train        Run current train.py once on GPU"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	@# Install only the local dev tools (modal CLI + anthropic SDK).
	@# torch==2.9.1+cu128 is Linux/CUDA only — it runs inside Modal, not here.
	uv pip install "modal>=0.73" "anthropic>=0.40.0"
	@uv run python -c "import modal, anthropic; print('✓ modal and anthropic installed')"

auth:
	@echo "Opening browser for Modal login..."
	uv run modal token new

secret:
	@read -p "Paste your Anthropic API key (sk-ant-...): " key; \
	uv run modal secret create anthropic ANTHROPIC_API_KEY=$$key
	@echo "✓ Secret 'anthropic' created in Modal"

# ── Verify ───────────────────────────────────────────────────────────────────

ping:
	uv run modal run modal_app.py::ping

check:
	uv run modal run modal_app.py::check_cache

# ── Data prep (one-time) ─────────────────────────────────────────────────────

prepare:
	@echo "Downloading data shards and training tokenizer on GPU..."
	@echo "This takes 15-30 minutes on first run. Safe to re-run (idempotent)."
	uv run modal run modal_app.py::prepare

# ── Experiments ──────────────────────────────────────────────────────────────

run:
	@echo "Launching autonomous agent loop: tag=$(RUN_TAG), experiments=$(N)"
	@echo "Modal streams logs here. Loop runs on Modal — safe to close terminal."
	uv run modal run modal_app.py --run-tag $(RUN_TAG) --n-experiments $(N)

run-detached:
	@echo "Launching detached — returns immediately, runs on Modal without this terminal."
	@echo "Check progress with: make results RUN_TAG=$(RUN_TAG)"
	uv run modal run --detach modal_app.py::agent_loop --run-tag $(RUN_TAG) --n-experiments $(N)

train:
	@echo "Running current train.py once on GPU..."
	uv run modal run modal_app.py::train

# ── Results ──────────────────────────────────────────────────────────────────

results:
	uv run modal run modal_app.py::show_results --run-tag $(RUN_TAG)

log:
	uv run modal run modal_app.py::show_log
