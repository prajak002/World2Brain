# Handoff: continuing on the Mac M1 (as of 2026-11-03)

Everything that is code, results JSON, figures, videos and the paper is in this repository on `main`. The two large
things that are not in git (checkpoints and episode data) are on Kaggle. This file tells a fresh machine, or a fresh
session, what exists, where it lives, what the numbers mean, and how to rerun each piece on Apple silicon.

## 1. Get the files

```bash
git clone https://github.com/prajak002/World2Brain.git && cd World2Brain
uv sync                                        # or: pip install -e . ; torch with MPS comes with the macOS wheel

# checkpoints (424 .pt, 5.8 GB) + every results folder from the 2026-11-02 GPU run
kaggle datasets download popsensharma/world2brain-run-2026-11-02 -p /tmp/run && unzip -o /tmp/run/*.zip -d .
# real phone-video episodes with V-JEPA teacher latents (274 MB); needed for anything "realvid"
kaggle datasets download popsensharma/world2brain-realvid -p /tmp/rv && unzip -o /tmp/rv/*.zip -d data/
```

The M1 already holds `data/*.npz` for the arena and the NYC twin, the FlyWire raw files and `data/connectome.npz`
(`~/worldmodel2connectom/data/`); copy them into `data/` if this is a new clone. The FlyWire-derived connectome file
is rebuilt by `python -m wm2cx.connectome` from the raw files if it is missing.

On the M1 pass `--device mps` to every script (the Windows box used `cuda`, the Vast box `cuda`, CPU works everywhere
but is slow for the teacher).

## 2. What is where

| path | what it is |
|---|---|
| `wm2cx/` | the library: `env.py` arena, `nyc.py` / `valdrada_env.py` city twin, `connectome.py` FlyWire subgraph + rewiring controls, `student.py` the leaky connectome-wired student and the GRU baselines, `teacher.py` V-JEPA 2.1 ViT-B encoder, `data.py` episode loading and PCA latents |
| `scripts/train.py`, `run_seeds.sh`, `aws_h100.sh` | training and the staged GPU pipeline (seeds, lr sweep, probes, planning, bench, ablation, real video, homing, goal-nav) |
| `scripts/eval_hd.py` | heading / position probes (ridge or MLP) with the action-only floor and the integrator ceiling |
| `scripts/eval_homing.py`, `eval_goalnav.py` | blind homing and goal-vector navigation from a linear readout of the state, with oracle / dead-reckoning / random controls |
| `scripts/eval_ckpt.py`, `ingest_video.py` | real-video transfer: ingest phone clips (egomotion by phase correlation), add teacher latents, zero-shot and leave-one-clip-out fine-tuned evaluation |
| `scripts/summarize.py` -> `results/summary.md` | every table in the paper is generated from the JSONs by this script |
| `results/hd`, `homing`, `goalnav`, `plan`, `bench`, `realvid_pilot`, `runs/*.json` | the numbers (JSON per run); `results/runs/*.pt` are the checkpoints (Kaggle) |
| `results/video/` | `wm2cx_paper.mp4` (arena: teacher vs connectome vs GRU through a blackout), `wm2cx_realvid.mp4` (same on a held-out phone clip), older explainer videos |
| `docs/results_draft_path_integration.md` | the prose results draft the paper was written from; `docs/explainer.html`, `brain3d.html` interactive explainers |
| `paper/` | the CVPR-format paper (`main.tex`, `sec/*.tex`, `main.bib`, `fig/`); build with Overleaf or `latexmk -pdf main.tex` |
| `scripts/make_paper_figs.py`, `make_video_paper.py`, `make_video_realvid.py` | regenerate the paper figures and the two videos from `results/` and `data/` |
| `PAPER_PLAN.md`, `PAPER_ISSUES.md` | the original plan and the reviewer-style issue register that drove the controls |
| `box_scripts/` | the shell scripts that ran on the Vast box (for provenance only) |

## 3. What the results say (so nobody re-derives them)

* **Prediction (E1).** Blind latent MSE on the arena, 5 seeds: GRU 0.434, recurrent-budget GRU 0.501, dense 0.562,
  connectome 0.579, random 0.583, block-shuffled 0.585, degree-shuffled 0.610. The wiring is *not* a better predictor.
* **Path integration (E3, the main result).** Heading decoded from the state during 8 dark steps: connectome 9.1 deg,
  block-shuffled 10.8, random 12.9, dense 21.3, GRU 30.6, budget GRU 35.5; action-count linear floor 40.1. Position R2
  0.94 vs 0.41 (GRU). Holds on the NYC twin, over 16 and 20 dark steps, and under an MLP probe. The real wiring beats
  block-shuffled by 2 to 6 deg; sign structure matters (unsigned 13.8) but not which signs (random Dale signs 9.2).
* **Behaviour (E9/E10).** Homing from a linear readout: arena 10 percent success (ceiling 100, floor 0), NYC 37 percent
  (block-shuffled 47, GRU 12, ceiling 92). Goal-vector navigation from the state equals random actions for every wiring.
  Both are reported as weak / negative in the paper.
* **Real video (E7).** Fine-tuned students reach MSE 1.18 to 1.26 but copy-last-visible scores 0.84; the heading readout
  is at the action-count floor. Negative, stated in the paper with the reasons (straight walking, 2 Hz).
* **Honest framing already agreed with the author.** The paper claims: connectome-scale sparse signed recurrence
  trained only on latent prediction forms a path-integrating state that a GRU does not; the real synapse list is a
  small consistent bonus. It does not claim better prediction, a ring attractor, working blind navigation or real-video
  transfer. This is not a CVPR-spotlight-grade result; it fits a neuro-AI venue.

## 4. Rerunning the key pieces on the M1

```bash
export PYTHONPATH=.
python scripts/summarize.py                                        # tables -> results/summary.md
python scripts/make_paper_figs.py                                  # paper/fig/{schematic,main_result,homing}.pdf
python scripts/make_video_paper.py --episode -1 --device mps       # arena video + paper/fig/video_strip.pdf (~3 min)
python scripts/make_video_realvid.py --device mps                  # real-clip video + paper/fig/realvid_strip.pdf (~3 min)
python scripts/eval_hd.py --run connectome_K1_s0 --device mps      # one probe run, writes results/hd/connectome_K1_s0.json
python scripts/eval_homing.py --run connectome_K1_s0 --readout state --device mps
```

To add new phone clips: `python scripts/ingest_video.py --videos a.mp4 b.mp4 --out data/realvid_new.npz --device mps`
(the teacher runs inside; ~70 s per 24-step episode on a small GPU, faster on the M1 GPU for ViT-B).

## 5. Known gaps and the next experiments

1. A behavioural readout that is not linear (MLP or recurrent readout fit on more rollouts) is the obvious way to turn
   the heading code into reliable homing.
2. Continuous turn angles (`collect.py --turn_noise`) so that integration is non-trivial and the action-count floor
   stops being exact.
3. Real video with real turns, odometry ground truth and a higher step rate, or a simulator with real imagery
   (Habitat + Matterport), for a closed loop on real scenes. A recorded clip cannot react to actions, so the current
   real-video test is open-loop by construction.
4. Mechanism: spectral analysis of the trained recurrent matrix and lesions of its slow modes, to test why sparse
   signed recurrence integrates.

## 6. Provenance of the environment used for this handoff

Windows laptop (RTX 3050 4 GB, miniconda base now carries `torch 2.13.0+cu130`), Vast.ai RTX 4090 box for the main run
(destroyed 2026-11-03), Kaggle as the mirror for checkpoints and data. The V-JEPA weights come from the Hugging Face
repo `apiantonio/vjepa2.1-vit-base-384` and download on first use.

## 7. Next phase (2026-11-04)

Plan to retarget the paper at a computer-vision venue with real-scene closed-loop navigation on 3D Gaussian splat twins of the phone clips, plus the note for the student collaborator (tasks, baselines, reading list).
