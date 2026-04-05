# Modal Training Loop Implementation Summary

This document summarizes the Modal infrastructure implementation for autoresearch.

## Implementation Overview

This implementation provides a complete Modal-based solution for running the autoresearch training loop with GPU support, following the requirements from the problem statement.

## Files Created

### Core Infrastructure

1. **`modal_runner.py`** - Main Modal application
   - `setup()` function: One-time data preparation (runs `prepare.py`)
   - `train()` function: Single training run (runs `train.py`)
   - `agent_loop()` function: Basic autonomous loop infrastructure
   - Configures H100 GPU, persistent volume, and Python 3.10+ image

2. **`modal_config.py`** - Configuration settings
   - GPU type selection (H100, A100, L40S, L4, T4)
   - Timeout configurations
   - Volume and experiment settings
   - Easy customization point for users

3. **`modal_agent_utils.py`** - Utility functions
   - Git operations (commit, reset, current hash)
   - Metrics parsing from training output
   - Results logging to TSV
   - Experiment decision logic (keep/discard)
   - Claude API integration helper

4. **`modal_agent_loop.py`** - Full agent implementation
   - Complete autonomous loop with Claude API
   - Automated code modification and evaluation
   - Result tracking and persistence
   - Production-ready experiment runner

### User Interface

5. **`modal_cli.py`** - Command-line interface
   - Simplified commands: `setup`, `train`, `run`, `status`
   - Wrapper around Modal commands
   - User-friendly experiment launching

### Documentation

6. **`MODAL_SETUP.md`** - Complete setup guide
   - Architecture explanation
   - Detailed usage instructions
   - Cost estimates
   - Advanced customization
   - Troubleshooting

7. **`MODAL_QUICKSTART.md`** - Quick start guide
   - 5-step getting started
   - Common commands
   - Monitoring and debugging
   - Expected costs

8. **Updated `README.md`** - Added Modal section
   - Quick reference to Modal option
   - Links to detailed guides

## Key Features

### ✅ GPU Support
- H100 as primary (matches autoresearch testing)
- A100, L40S, L4, T4 as alternatives
- Configurable via `modal_config.py`

### ✅ Persistent Storage
- Modal Volume mounted at `/data`
- Stores:
  - Training data (from prepare.py)
  - Tokenizer artifacts
  - Git repository with branches
  - Experiment logs and results
  - Model checkpoints

### ✅ Separation of Setup and Iteration
- `setup()`: Runs `uv run prepare.py` once
- `train()`: Runs `uv run train.py` repeatedly
- Clean separation as specified

### ✅ Agent Loop Infrastructure
- Git branch management per run-tag
- Baseline establishment
- Iterative experiment loop
- Keep/discard logic based on val_bpb
- Results tracking in TSV format

### ✅ Claude Integration
- Full Claude API integration in `modal_agent_loop.py`
- Reads `program.md` for instructions
- Proposes improvements to `train.py`
- Automated evaluation and decision-making

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Modal Cloud                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────┐                                        │
│  │  Modal Image   │                                        │
│  │  - Python 3.10 │                                        │
│  │  - uv          │                                        │
│  │  - autoresearch│                                        │
│  │  - dependencies│                                        │
│  └────────────────┘                                        │
│          │                                                  │
│          ▼                                                  │
│  ┌────────────────────────────────────────────┐            │
│  │         GPU Functions                      │            │
│  ├────────────────────────────────────────────┤            │
│  │  setup()       - H100/A100                │            │
│  │  train()       - Runs train.py            │            │
│  │  agent_loop()  - Autonomous experiments    │            │
│  └────────────────────────────────────────────┘            │
│          │                                                  │
│          ▼                                                  │
│  ┌────────────────────────────────────────────┐            │
│  │      Persistent Volume: /data              │            │
│  ├────────────────────────────────────────────┤            │
│  │  - Training data shards                    │            │
│  │  - BPE tokenizer                           │            │
│  │  - Git repos (per run-tag)                 │            │
│  │  - results.tsv                             │            │
│  │  - Experiment logs                         │            │
│  └────────────────────────────────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         │
         │ API Calls
         ▼
┌─────────────────────┐
│   Anthropic API     │
│   (Claude)          │
│  - Code generation  │
│  - Improvement ideas│
└─────────────────────┘
```

## Workflow

### One-Time Setup
```bash
modal run modal_runner.py::setup
```
1. Builds Modal image with Python 3.10 + uv + dependencies
2. Runs `uv run prepare.py` on H100 GPU
3. Downloads training data to persistent volume
4. Trains BPE tokenizer
5. Commits volume

### Single Training Run
```bash
modal run modal_runner.py::train
```
1. Mounts persistent volume at `/data`
2. Sets `HOME=/data` for cache access
3. Runs `uv run train.py` for 5 minutes
4. Parses and returns metrics

### Autonomous Agent Loop
```bash
modal run modal_agent_loop.py --run-tag apr5
```
1. Creates git repo in `/data/experiments/apr5`
2. Runs baseline training
3. Loop (up to max_experiments):
   - Calls Claude API with current state
   - Applies proposed changes to train.py
   - Commits to git
   - Runs training
   - Parses val_bpb
   - If improved: keep (branch advances)
   - If worse: discard (git reset)
   - Logs to results.tsv
4. Commits volume periodically

## Usage Examples

### Basic Usage
```bash
# Setup
pip install modal
modal setup
modal run modal_runner.py::setup

# Single experiment
modal run modal_runner.py::train

# Agent loop (requires Anthropic API key)
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...
modal run modal_agent_loop.py --run-tag apr5
```

### Advanced Usage
```bash
# Custom GPU (edit modal_config.py first)
GPU_TYPE = "A100-80GB"

# Limit experiments
modal run modal_agent_loop.py --run-tag apr5 --max-experiments 50

# Monitor logs
modal app logs autoresearch --follow

# Download results
modal volume get autoresearch-data experiments/apr5/results.tsv ./results.tsv
```

## Compliance with Requirements

### ✅ Modal Setup
- [x] Uses Modal for training loop
- [x] Treats Daytona as dev workspace (not used for GPU training)
- [x] Single NVIDIA GPU (H100/A100)
- [x] Modal GPU selectors (H100, A100-80GB, etc.)

### ✅ Repository Structure
- [x] Keeps repo structure intact
- [x] Core files: prepare.py, train.py, program.md
- [x] Agent edits train.py only
- [x] Human steers via program.md

### ✅ Persistent Storage
- [x] Modal Volume for persistence
- [x] Stores: repo checkout, tokenizer, data, logs, outputs
- [x] Designed for repeated overnight experiments

### ✅ Setup vs Iteration
- [x] `uv sync` in image build
- [x] `uv run prepare.py` once (in setup())
- [x] `uv run train.py` repeatedly (in agent loop)

### ✅ Agent Loop
- [x] Claude Code can access repo + program.md
- [x] Agent mutates train.py only
- [x] Branch per run-tag
- [x] Experiment notes after each attempt
- [x] Persists: diff, metric, time, branch, checkpoint
- [x] Keep on val_bpb improvement, else reset

## Cost Estimates

Based on Modal GPU pricing (approximate):

| GPU | Cost/Hour | Cost/Experiment | 100 Experiments |
|-----|-----------|----------------|-----------------|
| H100 | $4-5 | $0.40 | $40 |
| A100-80GB | $2-3 | $0.20 | $20 |
| L40S | $1-2 | $0.10 | $10 |
| L4 | $0.50-1 | $0.05 | $5 |

Each experiment: ~5 min training + ~30 sec overhead

## Testing

To test the implementation (requires Modal account):

```bash
# 1. Install Modal
pip install modal

# 2. Authenticate
modal setup

# 3. Test setup
modal run modal_runner.py::setup

# 4. Test single training run
modal run modal_runner.py::train

# 5. Test agent loop (requires API key)
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...
modal run modal_agent_loop.py --run-tag test --max-experiments 5
```

## Next Steps

1. **Test deployment**: User needs Modal account to test
2. **Tune prompts**: Optimize Claude prompts in `modal_agent_utils.py`
3. **Add monitoring**: Integrate with Modal dashboard
4. **Cost tracking**: Add cost estimation per run
5. **Multi-agent**: Extend to support multiple agents in parallel

## Notes

- Modal account required for deployment
- Anthropic API key required for autonomous loop
- H100 recommended for best performance (as tested by autoresearch)
- Volume persists data across runs
- Safe to stop/restart agent loop (volume is persistent)
- Results saved even if session interrupted

## Support

- Modal documentation: https://modal.com/docs
- Autoresearch: https://github.com/karpathy/autoresearch
- Issues: File in this repository
