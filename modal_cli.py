#!/usr/bin/env python3
"""
Simplified CLI for running autoresearch on Modal.

Usage:
    python modal_cli.py setup              # One-time data prep
    python modal_cli.py train              # Single training run
    python modal_cli.py run --tag apr5     # Start agent loop
    python modal_cli.py status             # Check volume status
"""

import argparse
import subprocess
import sys


def run_modal_command(command: str, **kwargs):
    """Run a Modal command and stream output."""
    cmd = ["modal", "run", "modal_runner.py"]

    if command == "setup":
        cmd.append("::setup")
    elif command == "train":
        cmd.append("::train")
    elif command == "agent_loop":
        cmd.append("::agent_loop")
        if "tag" in kwargs:
            cmd.extend(["--run-tag", kwargs["tag"]])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def check_status():
    """Check Modal volume status."""
    print("Checking autoresearch-data volume...")
    subprocess.run(["modal", "volume", "ls", "autoresearch-data"])


def main():
    parser = argparse.ArgumentParser(
        description="Run autoresearch experiments on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup                 # Run one-time data preparation
  %(prog)s train                 # Run single training experiment
  %(prog)s run --tag apr5        # Start autonomous agent loop
  %(prog)s status                # Check volume status

For more information, see MODAL_SETUP.md
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Setup command
    subparsers.add_parser("setup", help="One-time data preparation (runs prepare.py)")

    # Train command
    subparsers.add_parser("train", help="Run single training experiment (runs train.py)")

    # Run command (agent loop)
    run_parser = subparsers.add_parser("run", help="Start autonomous agent loop")
    run_parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Run tag identifier (e.g., 'apr5')",
    )

    # Status command
    subparsers.add_parser("status", help="Check Modal volume status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "status":
        check_status()
    elif args.command == "run":
        run_modal_command("agent_loop", tag=args.tag)
    else:
        run_modal_command(args.command)


if __name__ == "__main__":
    main()
