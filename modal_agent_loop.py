"""
Full agent loop implementation for Modal.

This extends modal_runner.py with a complete, production-ready agent loop
that integrates with Claude API for autonomous experimentation.

Usage:
    modal run modal_agent_loop.py::agent_loop_with_claude --run-tag apr5
"""

import os
import subprocess
import time
from pathlib import Path

import modal

from modal_runner import app, image, volume, GPU_CONFIG
from modal_agent_utils import (
    call_claude_for_improvement,
    git_current_commit,
    initialize_results_tsv,
    read_results_history,
    run_single_experiment,
    run_training,
    parse_training_output,
    log_result_to_tsv,
)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/data": volume},
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("anthropic-api-key")],
)
def agent_loop_with_claude(
    run_tag: str,
    max_experiments: int = 100,
    min_improvement: float = 0.0001,
):
    """
    Full autonomous agent loop with Claude API integration.

    This is the complete implementation of the autoresearch workflow:
    1. Initialize experiment directory and git repo
    2. Run baseline training
    3. Loop:
       a. Call Claude API to propose code change
       b. Apply change and commit
       c. Run training
       d. Evaluate and keep/discard
       e. Log results
    4. Continue until max_experiments or manually stopped

    Args:
        run_tag: Experiment identifier (e.g., 'apr5')
        max_experiments: Maximum experiments to run
        min_improvement: Minimum BPB improvement to consider significant

    Returns:
        dict with final results and statistics
    """
    print("=" * 80)
    print(f"🚀 Starting Claude-powered agent loop: {run_tag}")
    print(f"   Max experiments: {max_experiments}")
    print("=" * 80)

    # Setup paths
    os.environ["HOME"] = "/data"
    repo_dir = f"/data/experiments/{run_tag}"
    os.makedirs(repo_dir, exist_ok=True)

    # Copy workspace
    subprocess.run(["cp", "-r", "/workspace/.", repo_dir], check=True)

    # Initialize git
    if not os.path.exists(f"{repo_dir}/.git"):
        subprocess.run(["git", "init"], cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "user.email", "agent@autoresearch.ai"], cwd=repo_dir)
        subprocess.run(["git", "config", "user.name", "AutoResearch Agent"], cwd=repo_dir)
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
        subprocess.run(["git", "commit", "-m", "Initial baseline"], cwd=repo_dir, check=True)

    # Create branch
    branch_name = f"autoresearch/{run_tag}"
    subprocess.run(["git", "checkout", "-b", branch_name], cwd=repo_dir, check=False)

    # Initialize results
    results_file = f"{repo_dir}/results.tsv"
    initialize_results_tsv(results_file)

    # Read program.md
    with open(f"{repo_dir}/program.md") as f:
        program_md = f.read()

    # Run baseline
    print("\n" + "=" * 80)
    print("📊 Running baseline experiment...")
    print("=" * 80)

    baseline_metrics, baseline_output = run_training(repo_dir)

    if "val_bpb" not in baseline_metrics:
        raise RuntimeError("Baseline run failed - no val_bpb metric")

    baseline_bpb = baseline_metrics["val_bpb"]
    baseline_vram = baseline_metrics["peak_vram_mb"] / 1024.0

    commit_hash = git_current_commit(repo_dir)
    log_result_to_tsv(results_file, commit_hash, baseline_bpb, baseline_vram, "keep", "baseline")

    print(f"\n✓ Baseline established:")
    print(f"  val_bpb: {baseline_bpb:.6f}")
    print(f"  peak_vram: {baseline_vram:.1f} GB")
    print(f"  commit: {commit_hash}")

    # Experiment loop
    best_bpb = baseline_bpb
    experiments_run = 0
    improvements = 0
    crashes = 0

    print("\n" + "=" * 80)
    print("🔬 Starting autonomous experimentation...")
    print("=" * 80)

    for i in range(max_experiments):
        experiments_run += 1

        print(f"\n--- Experiment {experiments_run}/{max_experiments} ---")
        print(f"Current best val_bpb: {best_bpb:.6f}")

        try:
            # Read current train.py
            with open(f"{repo_dir}/train.py") as f:
                current_train_py = f.read()

            # Read results history
            results_history = read_results_history(results_file)

            # Call Claude for improvement
            print("🤖 Calling Claude API for improvement proposal...")
            new_train_py, description = call_claude_for_improvement(
                current_train_py=current_train_py,
                program_md=program_md,
                results_history=results_history,
                best_bpb=best_bpb,
            )

            print(f"💡 Proposed change: {description}")

            # Run experiment
            kept, new_best_bpb, metrics = run_single_experiment(
                repo_dir=repo_dir,
                new_train_py=new_train_py,
                description=description,
                best_bpb=best_bpb,
                baseline_vram_gb=baseline_vram,
                results_tsv_path=results_file,
            )

            # Check for crash
            if "val_bpb" not in metrics:
                crashes += 1
                print(f"  ⚠️  Total crashes so far: {crashes}")
                continue

            # Update best if kept
            if kept:
                improvements += 1
                best_bpb = new_best_bpb
                print(f"  🎉 Improvement #{improvements}!")

            # Print progress
            print(f"\n📈 Progress:")
            print(f"  Experiments: {experiments_run}")
            print(f"  Improvements: {improvements}")
            print(f"  Crashes: {crashes}")
            print(f"  Best val_bpb: {best_bpb:.6f}")
            print(f"  Total improvement: {baseline_bpb - best_bpb:.6f}")

        except Exception as e:
            print(f"❌ Error in experiment {experiments_run}: {e}")
            print("Continuing to next experiment...")
            continue

        # Commit volume periodically
        if experiments_run % 10 == 0:
            print("\n💾 Committing volume...")
            volume.commit()

    # Final summary
    print("\n" + "=" * 80)
    print("🏁 Experiment loop complete!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Total experiments: {experiments_run}")
    print(f"  Improvements: {improvements}")
    print(f"  Crashes: {crashes}")
    print(f"  Baseline val_bpb: {baseline_bpb:.6f}")
    print(f"  Best val_bpb: {best_bpb:.6f}")
    print(f"  Total improvement: {baseline_bpb - best_bpb:.6f}")
    print(f"  Improvement %: {((baseline_bpb - best_bpb) / baseline_bpb * 100):.2f}%")
    print(f"\nResults saved to: {results_file}")

    # Commit final volume state
    volume.commit()

    return {
        "run_tag": run_tag,
        "baseline_bpb": baseline_bpb,
        "best_bpb": best_bpb,
        "total_improvement": baseline_bpb - best_bpb,
        "experiments_run": experiments_run,
        "improvements": improvements,
        "crashes": crashes,
        "results_file": results_file,
    }


@app.local_entrypoint()
def main(run_tag: str = None, max_experiments: int = 100):
    """
    Local entrypoint for Claude-powered agent loop.

    Usage:
        modal run modal_agent_loop.py --run-tag apr5
        modal run modal_agent_loop.py --run-tag apr5 --max-experiments 50
    """
    if not run_tag:
        run_tag = time.strftime("run_%b%d").lower()
        print(f"No run tag specified, using: {run_tag}")

    result = agent_loop_with_claude.remote(
        run_tag=run_tag,
        max_experiments=max_experiments,
    )

    print(f"\n{'=' * 80}")
    print("Final Results:")
    print(f"{'=' * 80}")
    for key, value in result.items():
        print(f"{key}: {value}")
