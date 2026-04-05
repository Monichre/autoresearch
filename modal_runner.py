"""
Modal-based runner for autoresearch experiments.

This script runs the autoresearch training loop on Modal with GPU support.
It handles:
- GPU-backed training with H100/A100 support
- Persistent storage for data, tokenizer, and experiment results
- One-time setup (prepare.py) and iterative training (train.py)
- Agent-driven experiment loop with branch management

Usage:
    modal run modal_runner.py::setup     # One-time data preparation
    modal run modal_runner.py::train     # Single training run
    modal run modal_runner.py::agent_loop  # Autonomous experiment loop
"""

import os
import subprocess
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal Configuration
# ---------------------------------------------------------------------------

# Create Modal app
app = modal.App("autoresearch")

# Define the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl")
    .run_commands(
        # Install uv
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        'echo "export PATH=/root/.local/bin:$PATH" >> /root/.bashrc',
    )
    .env({"PATH": "/root/.local/bin:$PATH"})
    .copy_local_file("pyproject.toml", "/workspace/pyproject.toml")
    .copy_local_file("uv.lock", "/workspace/uv.lock")
    .workdir("/workspace")
    .run_commands(
        # Install dependencies using uv
        "/root/.local/bin/uv sync",
    )
    .copy_local_file("prepare.py", "/workspace/prepare.py")
    .copy_local_file("train.py", "/workspace/train.py")
    .copy_local_file("program.md", "/workspace/program.md")
)

# Create persistent volume for data, tokenizer, and experiment results
volume = modal.Volume.from_name("autoresearch-data", create_if_missing=True)

# GPU configuration - H100 as primary, A100 as fallback
GPU_CONFIG = modal.gpu.H100(count=1)
# Alternative: modal.gpu.A100(count=1, size="80GB")

# ---------------------------------------------------------------------------
# Setup Function (One-time)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/data": volume},
    timeout=3600,  # 1 hour for data download
)
def setup():
    """
    One-time setup: download data and train tokenizer.
    This runs prepare.py and stores artifacts in the persistent volume.
    """
    print("=" * 80)
    print("Starting one-time setup (prepare.py)")
    print("=" * 80)

    # Set cache directory to persistent volume
    cache_dir = "/data/cache"
    os.environ["HOME"] = "/data"  # Redirect home to persistent volume

    # Run prepare.py
    result = subprocess.run(
        ["/root/.local/bin/uv", "run", "prepare.py"],
        cwd="/workspace",
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"prepare.py failed with exit code {result.returncode}")

    # Commit volume changes
    volume.commit()

    print("=" * 80)
    print("Setup complete! Data and tokenizer stored in persistent volume.")
    print("=" * 80)

    return {"status": "success", "cache_dir": cache_dir}


# ---------------------------------------------------------------------------
# Training Function (Single Run)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/data": volume},
    timeout=900,  # 15 minutes (5 min training + overhead)
)
def train(train_py_content: str = None):
    """
    Run a single training experiment.

    Args:
        train_py_content: Optional custom train.py content. If None, uses the default.

    Returns:
        dict with metrics: val_bpb, training_seconds, peak_vram_mb, etc.
    """
    print("=" * 80)
    print("Starting training run")
    print("=" * 80)

    # Set cache directory to persistent volume
    os.environ["HOME"] = "/data"

    # If custom train.py content provided, write it
    if train_py_content:
        with open("/workspace/train.py", "w") as f:
            f.write(train_py_content)
        print("Using custom train.py")

    # Run training
    result = subprocess.run(
        ["/root/.local/bin/uv", "run", "train.py"],
        cwd="/workspace",
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Parse metrics from output
    metrics = {}
    for line in result.stdout.split("\n"):
        if line.startswith("val_bpb:"):
            metrics["val_bpb"] = float(line.split(":")[1].strip())
        elif line.startswith("training_seconds:"):
            metrics["training_seconds"] = float(line.split(":")[1].strip())
        elif line.startswith("total_seconds:"):
            metrics["total_seconds"] = float(line.split(":")[1].strip())
        elif line.startswith("peak_vram_mb:"):
            metrics["peak_vram_mb"] = float(line.split(":")[1].strip())
        elif line.startswith("mfu_percent:"):
            metrics["mfu_percent"] = float(line.split(":")[1].strip())
        elif line.startswith("total_tokens_M:"):
            metrics["total_tokens_M"] = float(line.split(":")[1].strip())
        elif line.startswith("num_steps:"):
            metrics["num_steps"] = int(line.split(":")[1].strip())
        elif line.startswith("num_params_M:"):
            metrics["num_params_M"] = float(line.split(":")[1].strip())
        elif line.startswith("depth:"):
            metrics["depth"] = int(line.split(":")[1].strip())

    metrics["returncode"] = result.returncode
    metrics["crashed"] = result.returncode != 0

    print("=" * 80)
    print("Training complete!")
    print(f"Metrics: {metrics}")
    print("=" * 80)

    return metrics


# ---------------------------------------------------------------------------
# Agent Loop (Autonomous Experimentation)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/data": volume},
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("anthropic-api-key")],  # For Claude API
)
def agent_loop(run_tag: str, max_experiments: int = 100):
    """
    Autonomous agent loop that modifies train.py and runs experiments.

    This implements the core autoresearch workflow:
    1. Read program.md for instructions
    2. Modify train.py with an experimental idea
    3. Run training and evaluate results
    4. Keep changes if val_bpb improves, otherwise revert
    5. Repeat indefinitely

    Args:
        run_tag: Experiment run identifier (e.g., 'apr5')
        max_experiments: Maximum number of experiments to run

    Returns:
        dict with experiment history
    """
    print("=" * 80)
    print(f"Starting autonomous agent loop: {run_tag}")
    print(f"Max experiments: {max_experiments}")
    print("=" * 80)

    # Set up persistent storage
    os.environ["HOME"] = "/data"

    # Initialize git repo in persistent volume
    repo_dir = f"/data/experiments/{run_tag}"
    os.makedirs(repo_dir, exist_ok=True)

    # Copy workspace to experiment directory
    subprocess.run(["cp", "-r", "/workspace/.", repo_dir], check=True)

    # Initialize git if needed
    if not os.path.exists(f"{repo_dir}/.git"):
        subprocess.run(["git", "init"], cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "user.email", "agent@autoresearch.ai"], cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "user.name", "AutoResearch Agent"], cwd=repo_dir, check=True)
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
        subprocess.run(["git", "commit", "-m", "Initial baseline"], cwd=repo_dir, check=True)

    # Create experiment branch
    branch_name = f"autoresearch/{run_tag}"
    subprocess.run(["git", "checkout", "-b", branch_name], cwd=repo_dir, check=False)

    # Initialize results tracking
    results_file = f"{repo_dir}/results.tsv"
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")

    # Run baseline
    print("\n" + "=" * 80)
    print("Running baseline experiment...")
    print("=" * 80)

    baseline_result = subprocess.run(
        ["/root/.local/bin/uv", "run", "train.py"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )

    # Parse baseline metrics
    baseline_bpb = None
    baseline_vram = None
    for line in baseline_result.stdout.split("\n"):
        if line.startswith("val_bpb:"):
            baseline_bpb = float(line.split(":")[1].strip())
        elif line.startswith("peak_vram_mb:"):
            baseline_vram = float(line.split(":")[1].strip()) / 1024.0

    if baseline_bpb is None:
        raise RuntimeError("Failed to parse baseline val_bpb")

    # Record baseline
    commit_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    ).stdout.strip()

    with open(results_file, "a") as f:
        f.write(f"{commit_hash}\t{baseline_bpb:.6f}\t{baseline_vram:.1f}\tkeep\tbaseline\n")

    best_bpb = baseline_bpb
    print(f"Baseline val_bpb: {baseline_bpb:.6f}")

    # Experiment loop
    experiments = []

    # NOTE: This is a simplified loop. In a full implementation, you would:
    # 1. Call Claude API to read program.md and propose changes to train.py
    # 2. Apply the changes and commit
    # 3. Run training
    # 4. Compare results and keep/revert
    # 5. Repeat

    print("\n" + "=" * 80)
    print("Agent loop setup complete!")
    print("=" * 80)
    print("\nNOTE: Full agent integration requires:")
    print("1. Claude API integration for code modifications")
    print("2. Experiment proposal logic based on program.md")
    print("3. Automated keep/discard decision making")
    print("\nThis function provides the infrastructure for running such a loop.")
    print("See program.md for the full experiment workflow.")

    return {
        "run_tag": run_tag,
        "baseline_bpb": baseline_bpb,
        "baseline_vram_gb": baseline_vram,
        "experiments_run": len(experiments),
        "best_bpb": best_bpb,
        "results_file": results_file,
    }


# ---------------------------------------------------------------------------
# Local Entrypoint (for testing)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(command: str = "train", run_tag: str = None):
    """
    Local entrypoint for running Modal functions.

    Usage:
        modal run modal_runner.py --command setup
        modal run modal_runner.py --command train
        modal run modal_runner.py --command agent_loop --run-tag apr5
    """
    if command == "setup":
        result = setup.remote()
        print(f"\nSetup result: {result}")

    elif command == "train":
        result = train.remote()
        print(f"\nTraining result: {result}")

    elif command == "agent_loop":
        if not run_tag:
            run_tag = time.strftime("run_%b%d").lower()
        result = agent_loop.remote(run_tag=run_tag)
        print(f"\nAgent loop result: {result}")

    else:
        print(f"Unknown command: {command}")
        print("Valid commands: setup, train, agent_loop")
