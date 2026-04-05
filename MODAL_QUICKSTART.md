# Quick Start: Running Autoresearch on Modal

This guide gets you up and running with autoresearch on Modal in under 10 minutes.

## Prerequisites

```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal setup
```

## Step 1: Configure GPU (Optional)

The default configuration uses H100 GPUs. To use a different GPU type, edit `modal_config.py`:

```python
GPU_TYPE = "A100-80GB"  # or "L40S", "L4", "T4"
```

## Step 2: Set up Anthropic API Key (Optional, for agent loop)

```bash
# Create Modal secret with your Anthropic API key
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...
```

Skip this if you only want to run manual experiments (not the autonomous agent loop).

## Step 3: Run One-Time Setup

This downloads training data and builds the tokenizer (~10-20 minutes):

```bash
modal run modal_runner.py::setup
```

## Step 4: Test with a Single Training Run

Run one 5-minute training experiment:

```bash
modal run modal_runner.py::train
```

You should see output like:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

## Step 5: Run Autonomous Agent Loop

Start the autonomous experiment loop that will run overnight:

```bash
# Simple version (infrastructure only)
modal run modal_runner.py::agent_loop --run-tag apr5

# Full Claude-powered version (requires API key)
modal run modal_agent_loop.py --run-tag apr5 --max-experiments 100
```

This will:
- Run baseline training
- Loop autonomously:
  - Call Claude API to propose improvements
  - Modify `train.py`
  - Run training
  - Keep if improved, discard if not
  - Log results

## Alternative: Use the CLI Helper

For convenience, use `modal_cli.py`:

```bash
# Setup
python modal_cli.py setup

# Single run
python modal_cli.py train

# Agent loop
python modal_cli.py run --tag apr5

# Check status
python modal_cli.py status
```

## Monitoring Your Experiments

### View logs in real-time

```bash
modal app logs autoresearch --follow
```

### Check volume contents

```bash
modal volume ls autoresearch-data
```

### Download results

```bash
# Get the results.tsv file
modal volume get autoresearch-data experiments/apr5/results.tsv ./results.tsv

# View the results
cat results.tsv
```

## Expected Costs

Based on Modal's GPU pricing (approximate):

- **H100**: ~$4-5/hour → ~$0.40 per 5-min experiment
- **A100-80GB**: ~$2-3/hour → ~$0.20 per experiment
- **L40S**: ~$1-2/hour → ~$0.10 per experiment
- **L4**: ~$0.50-1/hour → ~$0.05 per experiment

Overnight run (8 hours, ~96 experiments):
- **H100**: ~$32-40
- **A100**: ~$16-24
- **L40S**: ~$8-16
- **L4**: ~$4-8

## Troubleshooting

### "No module named 'modal'"

```bash
pip install modal
```

### "Volume not found"

Run `modal run modal_runner.py::setup` first to create the volume.

### "ANTHROPIC_API_KEY not found"

Either:
1. Skip the agent loop and run manual experiments
2. Create the secret: `modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...`

### "CUDA out of memory"

Switch to a larger GPU or reduce model size in `train.py`.

## Next Steps

- Read [MODAL_SETUP.md](MODAL_SETUP.md) for detailed documentation
- Customize `program.md` to guide the agent's research direction
- Review results in `results.tsv` after overnight runs
- Compare experiments and identify winning strategies

## Support

- Modal docs: https://modal.com/docs
- Autoresearch repo: https://github.com/karpathy/autoresearch
- Issues: https://github.com/Monichre/autoresearch/issues
