# Results log (append-only)

One entry per batch, newest last. Never edit an old entry; add a correction entry that points to it.

Template:

```
## <YYYY-MM-DD> <batch id> — <one line>
- commit: <git hash the run used>          box: <GPU, host alias>          wall time: <h>
- command: <exact command line, with env vars>
- prereg items: <H-ids from docs/PREREG.md>  -> PASS / FAIL / INCONCLUSIVE (with the number and its 95 % CI)
- outputs: results/<dir>/<files>          logs: logs/<file>
- failures / deviations from the runbook: <what and why; "none">
```

## 2026-11-10 setup — agent handoff docs, H200 launcher
- commit: see PR adding AGENTS.md / docs/PREREG.md / docs/RUNBOOK_GPU.md
- state: full-brain run connectome_K1_s0_full exists on the L4 (blind MSE 0.63 vs 0.58 subcircuit); the other
  full-brain runs and every external-benchmark run are still to do. The L4 box is stopped; its causal episode data
  have not yet been copied off.
