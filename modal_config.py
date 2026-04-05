"""
Modal configuration for autoresearch.

Customize GPU types, timeouts, and other settings here.
Import this in modal_runner.py to override defaults.
"""

# ---------------------------------------------------------------------------
# GPU Configuration
# ---------------------------------------------------------------------------

# Primary GPU configuration
# Tested on H100, as recommended by autoresearch README
GPU_TYPE = "H100"  # Options: H100, A100-80GB, A100-40GB, L40S, L4, T4

# GPU configurations by type
GPU_CONFIGS = {
    "H100": {"gpu": "H100", "count": 1},
    "A100-80GB": {"gpu": "A100", "count": 1, "size": "80GB"},
    "A100-40GB": {"gpu": "A100", "count": 1, "size": "40GB"},
    "L40S": {"gpu": "L40S", "count": 1},
    "L4": {"gpu": "L4", "count": 1},
    "T4": {"gpu": "T4", "count": 1},
}

# Get selected GPU config
def get_gpu_config():
    """Return Modal GPU configuration object."""
    import modal

    if GPU_TYPE == "H100":
        return modal.gpu.H100(count=1)
    elif GPU_TYPE == "A100-80GB":
        return modal.gpu.A100(count=1, size="80GB")
    elif GPU_TYPE == "A100-40GB":
        return modal.gpu.A100(count=1, size="40GB")
    elif GPU_TYPE == "L40S":
        return modal.gpu.L40S(count=1)
    elif GPU_TYPE == "L4":
        return modal.gpu.L4(count=1)
    elif GPU_TYPE == "T4":
        return modal.gpu.T4(count=1)
    else:
        raise ValueError(f"Unknown GPU type: {GPU_TYPE}")


# ---------------------------------------------------------------------------
# Timeout Configuration
# ---------------------------------------------------------------------------

# Timeouts in seconds
SETUP_TIMEOUT = 3600        # 1 hour for data download (prepare.py)
TRAIN_TIMEOUT = 900         # 15 minutes per experiment (5 min train + overhead)
AGENT_LOOP_TIMEOUT = 86400  # 24 hours for autonomous loop

# ---------------------------------------------------------------------------
# Volume Configuration
# ---------------------------------------------------------------------------

VOLUME_NAME = "autoresearch-data"
VOLUME_MOUNT_PATH = "/data"

# ---------------------------------------------------------------------------
# Python Configuration
# ---------------------------------------------------------------------------

PYTHON_VERSION = "3.10"  # Must be 3.10+ per autoresearch requirements

# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------

# Maximum number of experiments in agent loop
MAX_EXPERIMENTS = 100

# Default run tag format (uses current date)
import time
DEFAULT_RUN_TAG = time.strftime("run_%b%d").lower()

# Secret name for Anthropic API key (for Claude integration)
ANTHROPIC_SECRET_NAME = "anthropic-api-key"

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

# Git configuration for experiments
GIT_USER_EMAIL = "agent@autoresearch.ai"
GIT_USER_NAME = "AutoResearch Agent"

# Results tracking
RESULTS_FILENAME = "results.tsv"
RESULTS_COLUMNS = ["commit", "val_bpb", "memory_gb", "status", "description"]

# ---------------------------------------------------------------------------
# Advanced Configuration
# ---------------------------------------------------------------------------

# Whether to enable verbose logging
VERBOSE = True

# Number of CPU cores to allocate
CPU_COUNT = 4

# Memory allocation in GB
MEMORY_GB = 32

# Retry configuration for failed experiments
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 30
