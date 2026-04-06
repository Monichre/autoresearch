# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

---

## Research policy

### Objective

Find changes to `train.py` that reduce `val_bpb` on the fixed 5-minute training budget.
The search runs in phases. Complete each phase before moving to the next.
An agent on an H100 has no meaningful VRAM constraint — optimize purely for quality.

### Primary metric
`val_bpb` — lower is better. This is the only metric that determines keep vs. discard.

### Secondary metrics (tiebreakers only)
- `peak_vram_mb` — relevant only to flag regressions, not a constraint
- `mfu_percent` — useful signal: low MFU means the GPU is underutilized, not that the config is bad
- `num_params_M` — prefer smaller models at equal `val_bpb`
- Code simplicity — equal `val_bpb` with fewer lines is a keep

### Acceptance policy
- **Keep** if `val_bpb` drops by any amount, however small.
- **Keep tiny gains** (< 0.001) only if code is simpler or params are fewer.
- **Discard** anything that does not improve `val_bpb`.
- **Crash** = OOM, Python exception, or `val_bpb` > 100 / NaN. Attempt a fix once. If it fails again, log crash and move on.
- **Never combine two untested ideas** until each has been individually validated as a keep.

---

## Search phases

Work through these phases in order. Do not jump ahead.
When a phase produces 3 consecutive discards with no new ideas, advance to the next phase.

### Phase 1 — Throughput (start here)

Goal: maximize tokens processed in 5 minutes without hurting `val_bpb`.
More steps = more signal. Start here because throughput gains are free quality.

Allowed knobs:
- `DEVICE_BATCH_SIZE` — increase until MFU stops rising
- `TOTAL_BATCH_SIZE` — increase in powers of 2 (keep it a multiple of `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`)
- `DEPTH` — try shallower models that fit more steps
- `ASPECT_RATIO` / `HEAD_DIM` — try narrower models at same depth

Use tag `[throughput]` in description.

### Phase 2 — Optimizer

Goal: improve convergence rate under the 5-minute budget.

Allowed knobs:
- `MATRIX_LR`, `EMBEDDING_LR`, `UNEMBEDDING_LR`, `SCALAR_LR`
- `ADAM_BETAS`
- `WEIGHT_DECAY`
- `WARMUP_RATIO`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`
- Muon `ns_steps`, `beta2`, momentum schedule in `get_muon_momentum()`

Do not change architecture in this phase. One optimizer knob per experiment.
Use tag `[optimizer]` in description.

### Phase 3 — Architecture

Goal: find structural changes that improve sample efficiency.

Allowed knobs:
- `WINDOW_PATTERN` — try `SSSSL`, `SL`, `SSLL`, `L` (all long), `S` (all short)
- Value embedding coverage — try enabling or disabling VE on different layer indices (edit `has_ve()`)
- `n_kv_head` — try grouped-query attention (set lower than `n_head`)
- Activation function in `MLP.forward()` — try `F.gelu`, `F.silu` instead of `relu().square()`
- Norm placement — try pre-norm vs. post-norm
- Softcap value in `GPT.forward()` — try 10, 20, 30

Forbidden in this phase: changing optimizer settings (keep Phase 2 winners fixed).
Use tag `[arch]` in description.

### Phase 4 — Combine winners

Goal: stack the best individual changes from Phases 1-3.

Rules:
- Only combine changes that were individually validated as keeps.
- Combine at most 2 changes per experiment.
- If a combination is worse than either change alone, discard and do not retry that pair.

Use tag `[combo]` in description.

### Phase 5 — Free search

Goal: explore ideas not covered above.

Examples: custom LR schedules, mixed precision tweaks, alternative residual connections,
label smoothing, gradient clipping, GC management changes.

Use tag `[free]` in description.

---

## Forbidden edits (always)
- Do not modify `prepare.py` — it contains the fixed evaluation harness.
- Do not change `evaluate_bpb()`, `make_dataloader()`, `Tokenizer`, `MAX_SEQ_LEN`, or `TIME_BUDGET`.
- Do not add new dependencies or imports beyond what is already in `pyproject.toml`.
- Do not make compound edits: one change per experiment, always.

---

## Logging

Use tab-separated descriptions so `results.tsv` is analyzable. Format:

```
[tag] short description of what changed
```

Examples:
```
[throughput] increase DEVICE_BATCH_SIZE 128->256
[optimizer] MATRIX_LR 0.04->0.06
[arch] WINDOW_PATTERN SSSL->SSSSL
[combo] DEVICE_BATCH_SIZE 256 + MATRIX_LR 0.06
[free] cosine LR schedule instead of linear warmdown
```

---

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas in the current phase, advance to the next phase. If you complete all phases, re-run Phase 5 with new ideas.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

---

## Output format

Once the script finishes it prints a summary like this:

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

You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

---

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description with tag: `[throughput] ...`, `[optimizer] ...`, etc.

---

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Identify the current phase from `results.tsv` history
3. Tune `train.py` with an experimental idea by directly hacking the code
4. git commit
5. Run the experiment: `uv run train.py > run.log 2>&1`
6. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
10. If val_bpb is equal or worse, you git reset back to where you started
11. Check phase advancement rules: 3 consecutive discards with no new ideas → next phase
