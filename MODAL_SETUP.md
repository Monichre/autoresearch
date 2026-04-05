# Modal Setup Guide for Autoresearch

This guide explains how to run autoresearch experiments on Modal with GPU support.

## Overview

Modal provides GPU-backed infrastructure for running the autoresearch training loop. This setup:

- **Uses H100 GPUs** (or A100 as fallback) as recommended by the autoresearch repo
- **Persists data** across runs using Modal volumes
- **Separates setup from iteration**: runs `prepare.py` once, then `train.py` repeatedly
- **Supports autonomous agent loops** for overnight experimentation

## Prerequisites

1. **Modal account**: Sign up at https://modal.com
2. **Modal CLI**: Install and authenticate

```bash
pip install modal
modal setup
```

3. **API keys** (optional, for agent loop):
   - Anthropic API key for Claude integration
   - Store in Modal secrets: `modal secret create anthropic-api-key`

## Architecture

### Files

- **`modal_runner.py`**: Main Modal application with three functions:
  - `setup()`: One-time data preparation (runs `prepare.py`)
  - `train()`: Single training run (runs `train.py`)
  - `agent_loop()`: Autonomous experiment loop

### Modal Components

1. **Image**: Python 3.10 + uv + dependencies + autoresearch code
2. **GPU**: H100 (or A100-80GB) for training
3. **Volume**: Persistent storage mounted at `/data` for:
   - Downloaded training data
   - Trained tokenizer
   - Experiment results and logs
   - Git repository with experiment branches

## Quick Start

### Step 1: One-time Setup

Run data preparation and tokenizer training (takes ~10-20 minutes):

```bash
modal run modal_runner.py::setup
```

This downloads the training data and trains the BPE tokenizer, storing everything in a persistent Modal volume.

### Step 2: Run a Single Training Experiment

Run one 5-minute training experiment:

```bash
modal run modal_runner.py::train
```

Returns metrics:
```python
{
    'val_bpb': 0.997900,
    'training_seconds': 300.1,
    'peak_vram_mb': 45060.2,
    'mfu_percent': 39.80,
    'total_tokens_M': 499.6,
    'num_steps': 953,
    'num_params_M': 50.3,
    'depth': 8,
    'crashed': False
}
```

### Step 3: Run Autonomous Agent Loop

Run the full autonomous experiment loop:

```bash
modal run modal_runner.py::agent_loop --run-tag apr5
```

This will:
1. Create a git branch `autoresearch/apr5`
2. Run baseline training
3. Log results to `results.tsv`
4. (In full implementation) Iterate with agent modifications

## Customizing GPU Selection

Edit `modal_runner.py` to change GPU type:

```python
# Current: H100
GPU_CONFIG = modal.gpu.H100(count=1)

# Alternative options:
GPU_CONFIG = modal.gpu.A100(count=1, size="80GB")
GPU_CONFIG = modal.gpu.A100(count=1, size="40GB")
GPU_CONFIG = modal.gpu.L40S(count=1)
GPU_CONFIG = modal.gpu.L4(count=1)
GPU_CONFIG = modal.gpu.T4(count=1)
```

## Understanding the Workflow

### Data Flow

```
Local → Modal Image → Modal Volume
  ↓
pyproject.toml, train.py, prepare.py
  ↓
Modal builds image with uv sync
  ↓
setup() downloads data to /data volume
  ↓
train() uses /data volume for cache
  ↓
Results persisted in volume
```

### Experiment Loop Flow

```
1. Clone repo to /data/experiments/{run_tag}/
2. Initialize git repo and branch
3. Run baseline training
4. Record in results.tsv
5. Loop:
   - Modify train.py (agent proposes change)
   - git commit
   - Run training
   - Parse metrics
   - Compare val_bpb
   - If improved: keep (advance branch)
   - If worse: discard (git reset)
   - Record in results.tsv
```

## Cost Optimization

### GPU Costs (approximate, check Modal pricing)

- **H100**: ~$4-5/hour
- **A100 80GB**: ~$2-3/hour
- **L40S**: ~$1-2/hour
- **L4/T4**: ~$0.50-1/hour

### Recommendations

1. **Use H100 for production runs**: Fastest training, best for overnight experiments
2. **Use A100 for development**: Good balance of cost and performance
3. **Use L4/T4 for testing**: Cheapest for debugging Modal setup

### Time Budget

Each experiment is **5 minutes of training** + ~30 seconds startup/eval:
- **H100**: ~12 experiments/hour ≈ $0.40/experiment
- **A100**: ~12 experiments/hour ≈ $0.20/experiment
- **L4**: ~12 experiments/hour ≈ $0.08/experiment

Overnight (8 hours): ~96 experiments

## Advanced: Agent Integration

The `agent_loop()` function provides infrastructure for autonomous experimentation. To fully integrate with an AI agent (Claude, GPT-4, etc.):

1. **Read `program.md`** for experiment instructions
2. **Call agent API** to propose changes to `train.py`
3. **Apply changes** and commit to git
4. **Run training** using Modal
5. **Parse results** and decide keep/discard
6. **Loop** until max_experiments reached or stopped

Example integration pseudocode:

```python
# Inside agent_loop()
for i in range(max_experiments):
    # 1. Read current train.py and program.md
    with open(f"{repo_dir}/train.py") as f:
        current_code = f.read()

    # 2. Call agent API
    proposed_change = call_claude_api(
        current_code=current_code,
        program_md=program_md,
        previous_results=results_history
    )

    # 3. Apply changes
    with open(f"{repo_dir}/train.py", "w") as f:
        f.write(proposed_change.new_code)

    # 4. Commit
    subprocess.run(["git", "add", "train.py"], cwd=repo_dir)
    subprocess.run(["git", "commit", "-m", proposed_change.description], cwd=repo_dir)

    # 5. Run training
    result = subprocess.run(["/root/.local/bin/uv", "run", "train.py"], ...)

    # 6. Evaluate and keep/discard
    new_bpb = parse_val_bpb(result.stdout)
    if new_bpb < best_bpb:
        # Keep - branch advances automatically
        best_bpb = new_bpb
        status = "keep"
    else:
        # Discard - revert
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=repo_dir)
        status = "discard"

    # 7. Log
    log_to_tsv(commit, new_bpb, vram, status, description)
```

## Monitoring

### Check Volume Contents

```bash
modal volume ls autoresearch-data
```

### View Logs

Modal automatically captures stdout/stderr. View in Modal dashboard or:

```bash
modal app logs autoresearch
```

### Download Results

```bash
# Get results.tsv from volume
modal volume get autoresearch-data experiments/apr5/results.tsv ./results.tsv
```

## Troubleshooting

### "Volume not found"

First run needs to create the volume. Run `setup()` first.

### "CUDA out of memory"

- Reduce model size in `train.py` (decrease `DEPTH`)
- Use smaller GPU (T4/L4) for testing
- Check `DEVICE_BATCH_SIZE` in train.py

### "Prepare.py data download fails"

- Check internet connectivity
- Retry `setup()` - it resumes from where it left off
- Increase timeout in `@app.function(timeout=3600)`

### "Agent loop not modifying code"

- The provided `agent_loop()` is infrastructure only
- Full agent integration requires adding LLM API calls
- See "Advanced: Agent Integration" section above

## Next Steps

1. **Test the setup**: Run `modal run modal_runner.py::setup` and `train()`
2. **Verify results**: Check that metrics match local runs
3. **Add agent integration**: Implement Claude API calls in `agent_loop()`
4. **Run overnight**: Launch `agent_loop()` before bed
5. **Review results**: Check `results.tsv` in the morning

## References

- [Modal Documentation](https://modal.com/docs)
- [Modal GPU Support](https://modal.com/docs/guide/gpu)
- [Autoresearch README](README.md)
- [Autoresearch program.md](program.md)
