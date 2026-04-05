# Modal Setup Checklist

Use this checklist to verify your Modal setup for autoresearch.

## Prerequisites

- [ ] Modal account created at https://modal.com
- [ ] Modal CLI installed: `pip install modal`
- [ ] Modal authenticated: `modal setup`
- [ ] (Optional) Anthropic API key for agent loop

## Installation

- [ ] Modal dependency added to project
- [ ] Optional: Anthropic package installed: `uv sync --extra modal`

## Initial Setup

- [ ] Run setup function: `modal run modal_runner.py::setup`
- [ ] Verify volume created: `modal volume ls autoresearch-data`
- [ ] Check data downloaded successfully (should see confirmation in logs)
- [ ] Verify tokenizer trained (check setup logs)

## Testing

### Test Single Training Run

- [ ] Run: `modal run modal_runner.py::train`
- [ ] Check for metrics in output:
  - [ ] `val_bpb` present
  - [ ] `training_seconds` ≈ 300
  - [ ] `peak_vram_mb` > 0
  - [ ] No errors or crashes
- [ ] Verify GPU used (check Modal dashboard)

### Test Basic Agent Loop (Infrastructure)

- [ ] Run: `modal run modal_runner.py::agent_loop --run-tag test`
- [ ] Check baseline established
- [ ] Verify `results.tsv` created in volume
- [ ] Confirm branch created in git

### Test Full Agent Loop (with Claude)

- [ ] Create Anthropic secret: `modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...`
- [ ] Run: `modal run modal_agent_loop.py --run-tag test --max-experiments 5`
- [ ] Verify Claude API called successfully
- [ ] Check experiments run (at least 2-3)
- [ ] Confirm results logged to TSV
- [ ] Verify keep/discard logic working

## Monitoring

- [ ] Can view logs: `modal app logs autoresearch --follow`
- [ ] Can check volume: `modal volume ls autoresearch-data`
- [ ] Can download results: `modal volume get autoresearch-data experiments/test/results.tsv ./results.tsv`

## Production Readiness

- [ ] GPU type configured correctly (H100 recommended)
- [ ] Understand costs (see MODAL_QUICKSTART.md)
- [ ] `program.md` customized for your research goals
- [ ] API rate limits checked (Anthropic)
- [ ] Monitoring/alerting set up (optional)

## Troubleshooting

Common issues resolved:

- [ ] "No module named 'modal'" → Run `pip install modal`
- [ ] "Volume not found" → Run `setup()` first
- [ ] "ANTHROPIC_API_KEY not found" → Create Modal secret
- [ ] "CUDA out of memory" → Use larger GPU or reduce model size
- [ ] "prepare.py failed" → Check internet connection, retry

## Ready to Launch

When all items above are checked:

```bash
# Launch overnight run with your chosen run tag
modal run modal_agent_loop.py --run-tag apr5 --max-experiments 100
```

Expected behavior:
- Runs for ~8 hours (overnight)
- Completes ~96 experiments (12/hour × 8 hours)
- Results in `experiments/apr5/results.tsv`
- Best model kept in git branch `autoresearch/apr5`

## Post-Run Analysis

After overnight run completes:

- [ ] Download results: `modal volume get autoresearch-data experiments/apr5/results.tsv ./results.tsv`
- [ ] Review experiments: `cat results.tsv | column -t -s $'\t'`
- [ ] Check improvements: Count "keep" vs "discard" vs "crash"
- [ ] Analyze best changes: Look at git history in branch
- [ ] Update `program.md` based on learnings

## Notes

- Volume persists across runs - safe to stop/restart
- Each experiment costs ~$0.05-0.40 depending on GPU
- Claude API costs additional ~$0.01-0.05 per experiment
- Results are preserved even if session is interrupted

## Support

If you encounter issues:

1. Check MODAL_SETUP.md for detailed troubleshooting
2. Review Modal logs: `modal app logs autoresearch`
3. Check Modal dashboard for resource usage
4. File issue in repository with error logs
