"""
Modal GPU runner for karpathy/autoresearch — with autonomous agent loop.

Quick-start:
  modal run modal_app.py::prepare                                 # one-time data setup
  modal run modal_app.py::ping                               # verify connectivity (no GPU needed)
  modal run modal_app.py --run-tag apr5 --n-experiments 30       # launch autonomous loop
  modal run modal_app.py::show_results --run-tag apr5            # inspect results
  modal run modal_app.py::show_log                               # tail last train run

The agent loop (agent_loop) runs entirely on Modal:
  1. Reads current best train.py + experiment history
  2. Asks Claude to propose ONE targeted mutation
  3. Runs the mutation on a GPU via train_with_code()
  4. Keeps if val_bpb improves, discards otherwise
  5. Saves results to the autoresearch-workspace volume

Secrets required (create once in Modal dashboard or via CLI):
  modal secret create anthropic ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------

GPU           = "H100"   # "H100", "A100-80GB", "L40S", "L4", "T4"
NUM_SHARDS    = 10       # training shards to download; -1 = all 6542
GROQ_MODEL    = "llama-3.3-70b-versatile"  # swap to "qwen-qwq-32b" for deeper reasoning

REPO_DIR      = "/repo"
CACHE_DIR     = "/root/.cache/autoresearch"
RESULTS_DIR   = "/results"
WORKSPACE_DIR = "/workspace"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("autoresearch")

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

# Tokenizer + data shards (written by prepare, read by every training run).
data_volume = modal.Volume.from_name("autoresearch-cache", create_if_missing=True)

# Per-run logs (run.log from each training invocation).
results_volume = modal.Volume.from_name("autoresearch-results", create_if_missing=True)

# Experiment state: best train.py, results.tsv — one subdirectory per run_tag.
workspace_volume = modal.Volume.from_name("autoresearch-workspace", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    # PyTorch cu128 first — other packages link against it.
    .uv_pip_install(
        "torch==2.9.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install([
        "openai>=1.0.0",         # Gemini via OpenAI-compatible endpoint (primary LLM)
        "groq>=0.11.0",          # Groq fallback
        "kernels>=0.11.7",
        "matplotlib>=3.10.8",
        "numpy>=2.2.6",
        "pandas>=2.3.3",
        "pyarrow>=21.0.0",
        "requests>=2.32.0",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
    ])
    # Sync local repo into containers at startup (copy=False = no image rebuild on edits).
    # Modal 1.x replaces modal.Mount with add_local_dir on the image.
    # train.py edits are picked up on every function call without rebuilding.
    .add_local_dir(
        ".",
        remote_path=REPO_DIR,
        copy=False,
        ignore=[".git", ".venv", "__pycache__", "*.pyc", "*.pyo",
                "run.log", "results.tsv", "*.parquet"],
    )
)

# ---------------------------------------------------------------------------
# Internal helpers (run inside Modal containers, not locally)
# ---------------------------------------------------------------------------

def _parse_metrics(output: str, returncode: int) -> dict:
    """Parse the --- summary block from train.py stdout."""
    metrics: dict = {}
    in_summary = False
    for line in output.splitlines():
        if line.strip() == "---":
            in_summary = True
            continue
        if in_summary:
            m = re.match(r"^(\w+):\s+([\d.]+)", line)
            if m:
                try:
                    metrics[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    metrics["crashed"]    = returncode != 0 or "val_bpb" not in metrics
    metrics["returncode"] = returncode
    metrics["oom"]        = "OutOfMemoryError" in output or "out of memory" in output.lower()
    return metrics


def _apply_patch(original: str, patch_text: str) -> str:
    """
    Apply a patch produced by _call_llm to the original train.py.

    Expected patch format (delimiters on their own lines):
        <<<OLD>>>
        <exact lines to replace, copied verbatim>
        <<<NEW>>>
        <replacement lines>

    If the OLD block cannot be found verbatim, raises ValueError.
    """
    # Strip any accidental markdown fences the LLM might add
    patch_text = re.sub(r"```[a-z]*\n?", "", patch_text).strip()

    old_match = re.search(r"<<<OLD>>>\s*\n(.*?)(?=\n<<<NEW>>>)", patch_text, re.DOTALL)
    new_match = re.search(r"<<<NEW>>>\s*\n(.*?)$",               patch_text, re.DOTALL)

    if not old_match or not new_match:
        raise ValueError(f"Patch missing <<<OLD>>> or <<<NEW>>> block:\n{patch_text[:400]}")

    old_block = old_match.group(1)
    new_block = new_match.group(1).rstrip("\n")

    if old_block not in original:
        # Try stripping trailing whitespace on each line (common LLM slip)
        old_stripped = "\n".join(l.rstrip() for l in old_block.splitlines())
        orig_stripped = "\n".join(l.rstrip() for l in original.splitlines())
        if old_stripped not in orig_stripped:
            raise ValueError(
                f"<<<OLD>>> block not found in train.py.\n"
                f"OLD block (first 300 chars):\n{old_block[:300]}"
            )
        # Rebuild original without trailing whitespace so replace works
        original = "\n".join(l.rstrip() for l in original.splitlines())
        old_block = old_stripped

    return original.replace(old_block, new_block, 1)


def _call_llm(best_train_py: str, history_rows: list) -> tuple[str, str]:
    """
    Ask an LLM to propose ONE targeted mutation to train.py.

    Provider priority:
      1. Google Gemini 2.0 Flash  — if GOOGLE_API_KEY is set (1M tokens/day free)
      2. Groq llama-3.3-70b       — if GROQ_API_KEY is set (12k TPM free tier)

    Uses a patch format (<<<OLD>>>/<<<NEW>>>) so only the changed lines are
    returned, keeping requests well under token limits.

    Returns (modified_train_py, description).
    """
    import time
    from openai import OpenAI, RateLimitError, BadRequestError

    # Keep history small — each TSV row is ~60 tokens.
    history_block = (
        "\n".join(history_rows[-10:])
        if history_rows
        else "(none yet — this is the first mutation after baseline)"
    )

    system_prompt = (
        "You are an expert ML researcher running autonomous pretraining experiments.\n"
        "Goal: minimize val_bpb (bits per byte, lower is better) within a fixed "
        "5-minute GPU time budget on an H100 (80 GB VRAM).\n\n"
        "Hard rules:\n"
        "  - Change ONE thing per experiment. Never touch prepare.py.\n"
        "  - Do NOT modify evaluate_bpb(), make_dataloader(), Tokenizer, "
        "MAX_SEQ_LEN, or TIME_BUDGET.\n"
        "  - No new imports or packages beyond what is already in train.py.\n"
        "  - CRITICALLY: read the history below before proposing. NEVER retry an "
        "idea that already has status 'crash', 'discard', or 'skip'. "
        "If batch size increases caused OOM crashes, try SMALLER values or "
        "move to a different knob entirely.\n\n"
        "Search phases (follow in order, 3 consecutive non-keeps → advance):\n"
        "  Phase 1 [throughput]: DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE, DEPTH, ASPECT_RATIO\n"
        "  Phase 2 [optimizer]: LRs, ADAM_BETAS, WEIGHT_DECAY, warmup/warmdown\n"
        "  Phase 3 [arch]: WINDOW_PATTERN, has_ve(), n_kv_head, activation, norm\n"
        "  Phase 4 [combo]: combine individually validated wins only\n"
        "  Phase 5 [free]: anything else\n\n"
        "OUTPUT FORMAT — respond with EXACTLY this structure, nothing else:\n"
        "<<<DESCRIPTION>>>\n"
        "[phase] one-line description of the change\n"
        "<<<OLD>>>\n"
        "<copy the exact lines you are replacing, preserving indentation>\n"
        "<<<NEW>>>\n"
        "<replacement lines>\n\n"
        "Do NOT include markdown fences, explanations, or anything outside the three blocks."
    )

    user_message = (
        f"train.py:\n\n{best_train_py}\n\n"
        f"Experiment history (MUST read before proposing — avoid repeating crashes/discards):\n"
        f"{history_block}\n\n"
        "Output the patch for ONE NEW change (not in history) most likely to reduce val_bpb."
    )

    # Build provider list based on which API keys are available.
    # Each entry: (client, model_id, friendly_name)
    providers: list[tuple] = []

    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        providers.append((
            OpenAI(
                api_key=google_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            "gemini-2.0-flash",
            "Gemini 2.0 Flash",
        ))

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        from groq import Groq
        providers.append((
            Groq(api_key=groq_key),
            GROQ_MODEL,
            f"Groq/{GROQ_MODEL}",
        ))

    if not providers:
        raise RuntimeError(
            "No LLM API keys found. Set GOOGLE_API_KEY (recommended) or GROQ_API_KEY "
            "in Modal secrets. Run: modal secret create google GOOGLE_API_KEY=..."
        )

    def _try(client, model: str, name: str) -> tuple[str, str]:
        print(f"[_call_llm] provider={name}")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        raw = resp.choices[0].message.content.strip()
        desc_m = re.search(r"<<<DESCRIPTION>>>\s*\n(.*?)(?=\n<<<)", raw, re.DOTALL)
        description = desc_m.group(1).strip() if desc_m else "[free] LLM mutation"
        modified = _apply_patch(best_train_py, raw)
        return modified, description

    # Retry schedule: quick for per-minute limits, long for daily limits.
    def _wait(attempt: int) -> int:
        if attempt < 3:  return 70
        if attempt < 6:  return 1800
        return 3600

    MAX_RETRIES = 10   # up to ~10h wait before giving up on a provider

    for client, model, name in providers:
        for attempt in range(MAX_RETRIES):
            try:
                return _try(client, model, name)
            except RateLimitError as e:
                wait = _wait(attempt)
                print(f"[_call_llm] {name} rate-limited (attempt {attempt+1}), "
                      f"sleeping {wait}s...")
                time.sleep(wait)
            except BadRequestError as e:
                print(f"[_call_llm] {name} bad request — skipping: {e}")
                break
            except ValueError as e:
                print(f"[_call_llm] {name} patch parse error: {e}")
                break
        else:
            print(f"[_call_llm] {name} exhausted {MAX_RETRIES} retries, trying next provider")

    raise RuntimeError("All LLM providers exhausted. Check API keys and quotas.")


# ---------------------------------------------------------------------------
# prepare()  —  run once before any experiments
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    volumes={
        CACHE_DIR:   data_volume,
        RESULTS_DIR: results_volume,
    },
    timeout=3600,
)
def prepare(num_shards: int = NUM_SHARDS):
    """
    Download training shards and train the BPE tokenizer.
    Idempotent: already-present shards and a trained tokenizer are skipped.
    """
    env = {**os.environ, "PYTHONPATH": REPO_DIR}
    result = subprocess.run(
        [sys.executable, "prepare.py", "--num-shards", str(num_shards)],
        cwd=REPO_DIR,
        env=env,
    )
    data_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"prepare.py exited {result.returncode}")
    print(f"\nReady. {num_shards} shards + tokenizer in {CACHE_DIR}.")


# ---------------------------------------------------------------------------
# check_cache()  —  verify data before running experiments
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={CACHE_DIR: data_volume},
    timeout=30,
)
def check_cache() -> bool:
    """Print a summary of what's present in the cache volume."""
    data_dir = Path(CACHE_DIR) / "data"
    tok_dir  = Path(CACHE_DIR) / "tokenizer"
    shards   = sorted(data_dir.glob("shard_*.parquet")) if data_dir.exists() else []
    tok_pkl  = tok_dir / "tokenizer.pkl"
    tok_pt   = tok_dir / "token_bytes.pt"

    print(f"Data shards:    {len(shards)}")
    if shards:
        print(f"  first: {shards[0].name}  last: {shards[-1].name}")
    print(f"tokenizer.pkl:  {'OK' if tok_pkl.exists() else 'MISSING'}")
    print(f"token_bytes.pt: {'OK' if tok_pt.exists() else 'MISSING'}")

    ready = len(shards) >= 2 and tok_pkl.exists() and tok_pt.exists()
    print(f"\nReady: {'YES' if ready else 'NO — run prepare first'}")
    return ready


# ---------------------------------------------------------------------------
# ping()  —  lightweight connectivity / smoke-test (no GPU, no secrets)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={
        CACHE_DIR:     data_volume,
        RESULTS_DIR:   results_volume,
        WORKSPACE_DIR: workspace_volume,
    },
    timeout=30,
)
def ping() -> None:
    """
    Lightweight connectivity check. No GPU, no secrets.
    Run this first to confirm Modal, volumes, and image are working.
    Usage: modal run modal_app.py::ping
    """
    import platform
    # Import torch version as a plain string to avoid cloudpickle deserialization
    # issues when returning torch types to a local env without torch installed.
    try:
        import importlib.metadata
        torch_ver = importlib.metadata.version("torch")
    except Exception:
        torch_ver = "unknown"

    print(f"python:         {platform.python_version()}")
    print(f"platform:       {platform.system()}")
    print(f"torch:          {torch_ver}")
    print(f"cache vol:      {Path(CACHE_DIR).exists()}")
    print(f"results vol:    {Path(RESULTS_DIR).exists()}")
    print(f"workspace vol:  {Path(WORKSPACE_DIR).exists()}")
    print("OK")


# ---------------------------------------------------------------------------
# train_with_code()  —  GPU training with arbitrary train.py content
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    volumes={
        CACHE_DIR:   data_volume,
        RESULTS_DIR: results_volume,
    },
    timeout=900,   # 5 min training + torch.compile (~90s) + eval
)
def train_with_code(train_py_content: str) -> dict:
    """
    Run a candidate version of train.py on GPU and return its metrics.
    The content is written into the mounted /repo directory (in-container only;
    the local file is never modified).  prepare.py is available via the mount.
    """
    env = {**os.environ, "PYTHONPATH": REPO_DIR}
    log_path = Path(RESULTS_DIR) / "run.log"

    # Overwrite train.py inside the container (mount is writable in-container).
    (Path(REPO_DIR) / "train.py").write_text(train_py_content)

    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        env=env,
        timeout=850,
    )

    output = result.stdout + "\n" + result.stderr
    log_path.write_text(output)
    results_volume.commit()

    metrics = _parse_metrics(output, result.returncode)

    if metrics["crashed"]:
        tail = "\n".join(output.splitlines()[-30:])
        print(f"CRASH (exit={result.returncode})\n{tail}")
    else:
        vram = metrics.get("peak_vram_mb", 0) / 1024
        print(f"val_bpb={metrics['val_bpb']:.6f}  "
              f"vram={vram:.1f}GB  "
              f"mfu={metrics.get('mfu_percent', 0):.1f}%")

    return metrics


# ---------------------------------------------------------------------------
# train()  —  single run using local train.py (for manual experiments)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    volumes={
        CACHE_DIR:   data_volume,
        RESULTS_DIR: results_volume,
    },
    timeout=900,
)
def train() -> dict:
    """Run train.py as currently checked out locally. Returns metrics dict."""
    env = {**os.environ, "PYTHONPATH": REPO_DIR}
    log_path = Path(RESULTS_DIR) / "run.log"

    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        env=env,
        timeout=850,
    )
    output = result.stdout + "\n" + result.stderr
    log_path.write_text(output)
    results_volume.commit()

    metrics = _parse_metrics(output, result.returncode)
    if metrics["crashed"]:
        tail = "\n".join(output.splitlines()[-30:])
        print(f"CRASH (exit={result.returncode})\n{tail}")
    else:
        vram = metrics.get("peak_vram_mb", 0) / 1024
        print(f"val_bpb={metrics['val_bpb']:.6f}  vram={vram:.1f}GB  "
              f"mfu={metrics.get('mfu_percent', 0):.1f}%")
    return metrics


# ---------------------------------------------------------------------------
# agent_loop()  —  autonomous experiment loop (runs entirely on Modal)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={
        WORKSPACE_DIR: workspace_volume,
        CACHE_DIR:     data_volume,
        RESULTS_DIR:   results_volume,
    },
    secrets=[
        modal.Secret.from_name("google", required=False),  # Gemini (primary)
        modal.Secret.from_name("groq",   required=False),  # Groq (fallback)
    ],
    timeout=86400,   # 24 hours — runs overnight
)
def agent_loop(run_tag: str = "run", n_experiments: int = 20):
    """
    Autonomous keep/discard experiment loop:
      1. Establishes baseline (first call only)
      2. Asks Claude to mutate train.py
      3. Runs the mutation on GPU via train_with_code()
      4. Keeps if val_bpb improves, discards otherwise
      5. Persists state in autoresearch-workspace volume

    State is keyed by run_tag, so multiple sessions are independent.
    Re-running with the same run_tag resumes from where it left off.
    """
    # Per-session workspace
    ws = Path(WORKSPACE_DIR) / run_tag
    ws.mkdir(parents=True, exist_ok=True)

    best_py_path = ws / "train_py_best.py"
    tsv_path     = ws / "results.tsv"

    # --- Initialize on first call ---
    if not tsv_path.exists():
        # Seed best train.py from the local repo at invocation time
        best_py_path.write_text((Path(REPO_DIR) / "train.py").read_text())
        tsv_path.write_text("id\tval_bpb\tmemory_gb\tstatus\tdescription\n")
        workspace_volume.commit()

    best_train_py = best_py_path.read_text()

    # Recover best_bpb from existing history (handles resume)
    header, *rows = tsv_path.read_text().splitlines()
    best_bpb = None
    for row in reversed(rows):
        parts = row.split("\t")
        if len(parts) >= 4 and parts[3] == "keep":
            best_bpb = float(parts[1])
            break

    print(f"[agent_loop] run_tag={run_tag!r}  n_experiments={n_experiments}")
    print(f"[agent_loop] best_bpb={'(baseline pending)' if best_bpb is None else f'{best_bpb:.6f}'}")
    print(f"[agent_loop] experiments so far: {len(rows)}")

    # --- Baseline (only if no kept run exists) ---
    if best_bpb is None:
        print("[agent_loop] Running baseline...")
        m = train_with_code.remote(best_train_py)
        if m["crashed"]:
            raise RuntimeError("Baseline run crashed — fix train.py before launching agent_loop")
        best_bpb = m["val_bpb"]
        vram_gb  = m.get("peak_vram_mb", 0) / 1024
        rows.append(f"baseline\t{best_bpb:.6f}\t{vram_gb:.1f}\tkeep\tbaseline")
        _flush_tsv(tsv_path, header, rows, workspace_volume)
        print(f"[agent_loop] Baseline: val_bpb={best_bpb:.6f}")

    # --- Main loop ---
    for i in range(n_experiments):
        exp_id = f"exp{len(rows):03d}"
        print(f"\n[agent_loop] {exp_id} ({i+1}/{n_experiments}) | best={best_bpb:.6f}")

        # Generate mutation (returns modified train.py + description)
        try:
            candidate, exp_desc = _call_llm(best_train_py, rows)
        except Exception as e:
            print(f"[agent_loop] {exp_id} LLM/patch error: {e}")
            rows.append(f"{exp_id}\t0.000000\t0.0\tskip\tLLM error: {str(e)[:80]}")
            _flush_tsv(tsv_path, header, rows, workspace_volume)
            continue

        print(f"[agent_loop] {exp_id} proposed: {exp_desc}")

        # Run on GPU
        m = train_with_code.remote(candidate)
        vram_gb = m.get("peak_vram_mb", 0) / 1024

        if m["crashed"]:
            crash_tag = "OOM" if m.get("oom") else "error"
            print(f"[agent_loop] {exp_id} CRASH ({crash_tag})")
            rows.append(f"{exp_id}\t0.000000\t0.0\tcrash\t{exp_desc} [{crash_tag}]")
        elif m["val_bpb"] < best_bpb:
            print(f"[agent_loop] {exp_id} KEEP  {m['val_bpb']:.6f} < {best_bpb:.6f}")
            best_bpb      = m["val_bpb"]
            best_train_py = candidate
            best_py_path.write_text(best_train_py)
            rows.append(f"{exp_id}\t{m['val_bpb']:.6f}\t{vram_gb:.1f}\tkeep\t{exp_desc}")
        else:
            print(f"[agent_loop] {exp_id} DISCARD  {m['val_bpb']:.6f} >= {best_bpb:.6f}")
            rows.append(f"{exp_id}\t{m['val_bpb']:.6f}\t{vram_gb:.1f}\tdiscard\t{exp_desc}")

        _flush_tsv(tsv_path, header, rows, workspace_volume)

    print(f"\n[agent_loop] Complete.")
    print(f"  Best val_bpb : {best_bpb:.6f}")
    print(f"  Experiments  : {len(rows) - 1}")  # subtract baseline
    kept = sum(1 for r in rows if "\tkeep\t" in r and "baseline" not in r)
    print(f"  Kept         : {kept}")
    print(f"  Results TSV  : {tsv_path}")


def _flush_tsv(path: Path, header: str, rows: list, volume) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    volume.commit()


# ---------------------------------------------------------------------------
# show_results()  —  print experiment TSV for a run_tag
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={WORKSPACE_DIR: workspace_volume},
    timeout=30,
)
def show_results(run_tag: str = "run"):
    """Print results.tsv for the given run_tag."""
    tsv_path = Path(WORKSPACE_DIR) / run_tag / "results.tsv"
    if not tsv_path.exists():
        print(f"No results for run_tag={run_tag!r}. Check the run_tag or run prepare first.")
        return
    print(tsv_path.read_text())


# ---------------------------------------------------------------------------
# show_log()  —  tail the last training run log
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={RESULTS_DIR: results_volume},
    timeout=30,
)
def show_log(tail_lines: int = 60):
    """Print the tail of the most recent run.log."""
    log_path = Path(RESULTS_DIR) / "run.log"
    if not log_path.exists():
        print("No run log found.")
        return
    lines = log_path.read_text().splitlines()
    print("\n".join(lines[-tail_lines:]))


# ---------------------------------------------------------------------------
# Local entrypoint  —  `modal run modal_app.py [--run-tag X] [--n-experiments N]`
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(run_tag: str = "run", n_experiments: int = 20):
    """
    Launch the autonomous agent loop on Modal.

    Examples:
      modal run modal_app.py                                    # defaults
      modal run modal_app.py --run-tag apr5 --n-experiments 50
      modal run modal_app.py --run-tag apr5 --n-experiments 10  # resume
    """
    print(f"Launching agent_loop: run_tag={run_tag!r}, n_experiments={n_experiments}")
    print("Modal will stream logs here. The loop runs on Modal — safe to close terminal.\n")
    agent_loop.remote(run_tag=run_tag, n_experiments=n_experiments)
