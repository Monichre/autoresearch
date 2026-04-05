"""
Agent integration utilities for autoresearch on Modal.

This module provides helper functions for integrating AI agents
(Claude, GPT-4, etc.) with the autoresearch experiment loop.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Git Utilities
# ---------------------------------------------------------------------------

def git_current_commit(repo_dir: str) -> str:
    """Get current git commit hash (short)."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_commit_changes(repo_dir: str, message: str) -> str:
    """Commit changes to git and return commit hash."""
    subprocess.run(["git", "add", "train.py"], cwd=repo_dir, check=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_dir,
        check=True,
    )
    return git_current_commit(repo_dir)


def git_reset_to_previous(repo_dir: str) -> None:
    """Reset git to previous commit (discard changes)."""
    subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        cwd=repo_dir,
        check=True,
    )


# ---------------------------------------------------------------------------
# Metrics Parsing
# ---------------------------------------------------------------------------

def parse_training_output(stdout: str) -> Dict[str, float]:
    """
    Parse training script output to extract metrics.

    Returns dict with: val_bpb, training_seconds, peak_vram_mb, etc.
    Returns empty dict if parsing fails (crashed run).
    """
    metrics = {}

    for line in stdout.split("\n"):
        if ":" not in line:
            continue

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        try:
            if key == "val_bpb":
                metrics["val_bpb"] = float(value)
            elif key == "training_seconds":
                metrics["training_seconds"] = float(value)
            elif key == "total_seconds":
                metrics["total_seconds"] = float(value)
            elif key == "peak_vram_mb":
                metrics["peak_vram_mb"] = float(value)
            elif key == "mfu_percent":
                metrics["mfu_percent"] = float(value)
            elif key == "total_tokens_M":
                metrics["total_tokens_M"] = float(value)
            elif key == "num_steps":
                metrics["num_steps"] = int(value)
            elif key == "num_params_M":
                metrics["num_params_M"] = float(value)
            elif key == "depth":
                metrics["depth"] = int(value)
        except (ValueError, IndexError):
            continue

    return metrics


def check_crash(metrics: Dict[str, float]) -> bool:
    """Check if training run crashed (no val_bpb metric)."""
    return "val_bpb" not in metrics or metrics.get("val_bpb") == 0.0


# ---------------------------------------------------------------------------
# Results Logging
# ---------------------------------------------------------------------------

def log_result_to_tsv(
    tsv_path: str,
    commit: str,
    val_bpb: float,
    memory_gb: float,
    status: str,
    description: str,
) -> None:
    """
    Append result to results.tsv file.

    Args:
        tsv_path: Path to results.tsv
        commit: Git commit hash (short)
        val_bpb: Validation bits per byte
        memory_gb: Peak VRAM in GB
        status: 'keep', 'discard', or 'crash'
        description: Short description of experiment
    """
    # Sanitize description (remove tabs and newlines)
    description = description.replace("\t", " ").replace("\n", " ").strip()

    line = f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"

    with open(tsv_path, "a") as f:
        f.write(line)


def initialize_results_tsv(tsv_path: str) -> None:
    """Create results.tsv with header if it doesn't exist."""
    if not os.path.exists(tsv_path):
        with open(tsv_path, "w") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")


def read_results_history(tsv_path: str) -> List[Dict]:
    """
    Read results.tsv and return list of experiment results.

    Returns list of dicts with keys: commit, val_bpb, memory_gb, status, description
    """
    if not os.path.exists(tsv_path):
        return []

    results = []
    with open(tsv_path, "r") as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 5:
            results.append({
                "commit": parts[0],
                "val_bpb": float(parts[1]) if parts[1] != "0.000000" else None,
                "memory_gb": float(parts[2]),
                "status": parts[3],
                "description": parts[4],
            })

    return results


# ---------------------------------------------------------------------------
# Training Execution
# ---------------------------------------------------------------------------

def run_training(repo_dir: str, uv_path: str = "/root/.local/bin/uv") -> Tuple[Dict, str]:
    """
    Run training script and return metrics + full output.

    Returns:
        (metrics_dict, stdout_string)
    """
    result = subprocess.run(
        [uv_path, "run", "train.py"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )

    metrics = parse_training_output(result.stdout)
    metrics["returncode"] = result.returncode

    return metrics, result.stdout


# ---------------------------------------------------------------------------
# Claude API Integration (Example)
# ---------------------------------------------------------------------------

def call_claude_for_improvement(
    current_train_py: str,
    program_md: str,
    results_history: List[Dict],
    best_bpb: float,
    anthropic_api_key: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Call Claude API to propose an improvement to train.py.

    This is a simplified example. In production, you'd want more sophisticated
    prompting, context management, and error handling.

    Args:
        current_train_py: Current content of train.py
        program_md: Content of program.md (agent instructions)
        results_history: List of previous experiment results
        best_bpb: Best validation BPB achieved so far
        anthropic_api_key: Anthropic API key (from env if None)

    Returns:
        (new_train_py_content, description_of_change)
    """
    if anthropic_api_key is None:
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. "
            "Add 'anthropic>=0.18.0' to dependencies to use Claude integration."
        )

    client = anthropic.Anthropic(api_key=anthropic_api_key)

    # Format results history for context
    history_str = "\n".join([
        f"- {r['description']}: val_bpb={r['val_bpb']:.6f}, status={r['status']}"
        for r in results_history[-10:]  # Last 10 experiments
    ])

    prompt = f"""You are an AI research assistant working on autoresearch experiments.

Your goal: Modify train.py to achieve a lower val_bpb (validation bits per byte).

Current best val_bpb: {best_bpb:.6f}

Program instructions (from program.md):
{program_md}

Recent experiment history:
{history_str}

Current train.py:
```python
{current_train_py}
```

Propose ONE specific change to train.py that might improve val_bpb.
The change should be:
1. Small and focused (modify one thing at a time)
2. Based on solid ML principles
3. Within the 5-minute time budget
4. Unlikely to cause OOM errors

Respond with:
1. A brief description of the change (one sentence)
2. The complete modified train.py file

Format your response as:
DESCRIPTION: <one sentence>
CODE:
```python
<full modified train.py>
```
"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text

    # Parse response
    description_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|CODE:)", response_text, re.DOTALL)
    code_match = re.search(r"```python\n(.+?)\n```", response_text, re.DOTALL)

    if not description_match or not code_match:
        raise ValueError("Failed to parse Claude response")

    description = description_match.group(1).strip()
    new_code = code_match.group(1).strip()

    return new_code, description


# ---------------------------------------------------------------------------
# Experiment Decision Logic
# ---------------------------------------------------------------------------

def should_keep_change(
    new_bpb: float,
    best_bpb: float,
    new_vram_gb: float,
    baseline_vram_gb: float,
    vram_threshold_multiplier: float = 1.2,
) -> Tuple[bool, str]:
    """
    Decide whether to keep a change based on metrics.

    Args:
        new_bpb: New validation BPB
        best_bpb: Best BPB so far
        new_vram_gb: New peak VRAM in GB
        baseline_vram_gb: Baseline VRAM in GB
        vram_threshold_multiplier: Max acceptable VRAM increase (default 1.2x)

    Returns:
        (should_keep: bool, reason: str)
    """
    # Check if improved
    if new_bpb < best_bpb:
        improvement = best_bpb - new_bpb

        # Check VRAM constraint
        vram_ratio = new_vram_gb / baseline_vram_gb if baseline_vram_gb > 0 else 1.0

        if vram_ratio > vram_threshold_multiplier:
            return False, f"Improved BPB by {improvement:.6f} but VRAM increased {vram_ratio:.2f}x"

        return True, f"Improved BPB by {improvement:.6f}"

    elif new_bpb == best_bpb:
        return False, "No improvement in BPB"

    else:
        regression = new_bpb - best_bpb
        return False, f"BPB regressed by {regression:.6f}"


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    repo_dir: str,
    new_train_py: str,
    description: str,
    best_bpb: float,
    baseline_vram_gb: float,
    results_tsv_path: str,
    uv_path: str = "/root/.local/bin/uv",
) -> Tuple[bool, float, Dict]:
    """
    Run a single experiment: apply change, train, evaluate, keep or discard.

    Args:
        repo_dir: Path to git repository
        new_train_py: New content for train.py
        description: Description of change
        best_bpb: Best BPB so far
        baseline_vram_gb: Baseline VRAM in GB
        results_tsv_path: Path to results.tsv
        uv_path: Path to uv binary

    Returns:
        (kept: bool, new_bpb: float, metrics: dict)
    """
    # Write new train.py
    train_py_path = os.path.join(repo_dir, "train.py")
    with open(train_py_path, "w") as f:
        f.write(new_train_py)

    # Commit
    commit_hash = git_commit_changes(repo_dir, description)

    # Run training
    print(f"Running experiment: {description}")
    metrics, stdout = run_training(repo_dir, uv_path)

    # Check for crash
    if check_crash(metrics):
        print("  ❌ Experiment crashed")
        log_result_to_tsv(results_tsv_path, commit_hash, 0.0, 0.0, "crash", description)
        git_reset_to_previous(repo_dir)
        return False, best_bpb, metrics

    # Evaluate
    new_bpb = metrics["val_bpb"]
    new_vram_gb = metrics["peak_vram_mb"] / 1024.0

    keep, reason = should_keep_change(new_bpb, best_bpb, new_vram_gb, baseline_vram_gb)

    if keep:
        print(f"  ✓ Keeping change: {reason}")
        log_result_to_tsv(results_tsv_path, commit_hash, new_bpb, new_vram_gb, "keep", description)
        return True, new_bpb, metrics
    else:
        print(f"  ✗ Discarding change: {reason}")
        log_result_to_tsv(results_tsv_path, commit_hash, new_bpb, new_vram_gb, "discard", description)
        git_reset_to_previous(repo_dir)
        return False, best_bpb, metrics
