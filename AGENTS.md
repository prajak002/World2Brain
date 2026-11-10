# AGENTS.md — read this first (any coding agent: Claude Code, Astra, Codex, ...)

This repository is **World2Brain / wm2cx**: a video world model (V-JEPA 2.1 ViT-B, the *teacher*) is distilled into a
recurrent *student* whose recurrent wiring is the **whole proofread FlyWire v783 Drosophila brain** (139,248 neurons,
2,700,429 signed edges at >= 5 synapses). Controls: the same student with degree-shuffled, block-shuffled and
Erdos-Renyi wiring, and parameter-matched GRU / transformer students. Target venue: CVPR (abstract 2026-11-11,
paper 2026-11-16).

You are running GPU experiments for this paper on behalf of a PhD student. Work autonomously, in batches, and leave
every result reproducible. The four documents you need, in order:

| file | what it gives you |
|---|---|
| `docs/PREREG.md` | the **frozen** hypotheses, metrics, seeds, statistics and stopping rules. Do not change it after a run has started; if you must, add a dated amendment at the bottom and say why. |
| `docs/RUNBOOK_GPU.md` | the batch queue: what to run, on which box, exact commands, expected time and memory, and the acceptance check for each batch |
| `docs/HANDOFF_M1.md` | what every file and existing result is (so you do not re-derive them) |
| `docs/BENCHMARK_BLACKOUT.md` | the in-house Blackout benchmark (now the *secondary* evaluation; external benchmarks come first) |

## Ground rules (non-negotiable; these are what make the results hold up under review)

1. **Pre-registration first.** Every hypothesis, metric, seed list and comparison is in `docs/PREREG.md` before the
   run. Commit the pre-registration *before* launching; the commit hash is the registration timestamp.
2. **Never tune on test.** Learning rates, epochs and probe regularisation are chosen on a validation split cut from
   training data (`--val 0.15`). The test split is evaluated once per final configuration.
3. **Matched baselines.** Every comparison is at equal trainable parameters (+-5 %), equal data, equal epochs and
   equal compute budget. Report the parameter breakdown (`param_breakdown` in each run JSON).
4. **Seeds and statistics.** 5 seeds per condition for headline numbers (3 is the floor for ablations). Report mean,
   bootstrap 95 % CI over test episodes, paired tests against the connectome, Holm correction across the comparison
   family. A difference whose CI overlaps zero is reported as "no difference", not as a trend.
5. **External benchmarks use their published protocol unchanged** (splits, metrics, horizons, probe architecture).
   Quote the published baseline numbers next to ours *and* rerun at least one published baseline in our pipeline, so
   a pipeline bug cannot masquerade as a win.
6. **Report negatives.** A failed hypothesis goes into the results table and the paper; it is not dropped or rerun
   with new settings until it passes. Reruns with changed settings are new, separately labelled conditions.
7. **No claim without a number.** Never write "state of the art", "outperforms" or "significant" unless the JSON in
   `results/` supports it with the statistics of rule 4.
8. **Restartable and idempotent.** Every stage skips work whose output JSON already exists. Never delete or
   overwrite results, checkpoints or data; write new conditions under new run names.

## Environment

```bash
git clone https://github.com/prajak002/World2Brain.git && cd World2Brain
uv sync                                    # Python env from uv.lock; on AWS DLAMI boxes /opt/pytorch/bin/python also works
export PYTHONPATH=.                        # every script assumes it
uv run python -m pytest tests -q           # must pass before any batch
```

* Data is **not** in git (FlyWire is CC-BY-NC). Episode files (`data/*.npz`), the full-brain graph
  (`data/connectome_full.npz`) and checkpoints come from the lab's L4 box or Kaggle; see `docs/RUNBOOK_GPU.md` §0.
* Every script takes `--device cuda|mps|cpu`. Teacher weights download from Hugging Face
  (`apiantonio/vjepa2.1-vit-base-384`, see `wm2cx/teacher.py`).

## Where things live

* `wm2cx/` library (`connectome.py` graph + rewiring controls, `student.py` students, `teacher.py` V-JEPA encoder,
  `data.py` episodes + PCA latents, `env.py` / `nyc.py` simulators).
* `scripts/train.py` trains one run -> `results/runs/<run>.json` + `.pt`. Run name = `<wiring>_K1_s<seed><tag>`.
* `scripts/run_l4_full.sh` the full-brain pipeline (`JOBS=N` runs N trainings concurrently on a big GPU).
* `scripts/aws_l4_launch.sh` drives a remote ssh GPU box from a laptop (`push-full`, `run-full`, `log-full`, `pull`).
* `scripts/summarize.py` builds every paper table from the JSONs -> `results/summary.md`. Paper numbers come only
  from there (macros in `paper_v2/`).

## Working agreement

* Branch per batch (`exp/<batch>-<date>`), PR into `main`; never push to `main` directly. Commit messages carry no
  AI co-author trailers. Never commit `.env*`, keys, `.pem` files, data or checkpoints.
* Never edit a shell script while it is running (bash reads it lazily); copy it, or wait.
* After each batch, append a dated entry to `docs/RESULTS_LOG.md`: commit hash, box, command, wall time, what passed or
  failed against `docs/PREREG.md`, and links to the JSONs. Then run `scripts/summarize.py` and commit both.
* When something fails, keep the log (`logs/`), write the failure into `docs/RESULTS_LOG.md`, and fix it on a branch;
  do not silently retry with different hyperparameters.
* Stop paying: shut down rented or borrowed GPU boxes after pulling results.
