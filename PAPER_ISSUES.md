# Issues register: ConneJEPA / wm2cx vs. a CVPR spotlight bar

Reviewed 2026-10-30 against the mock manuscript ("ConneJEPA: Distilling Video World Models into Biological
Connectomes for Embodied Navigation") and the code + results in this repo (`wm2cx/`, `scripts/`, `results/`).
Severity: **P0** = blocks the claim as written, **P1** = a reviewer will reject on it, **P2** = fix before camera-ready.

Numbers below were recomputed from `data/connectome.npz`, `results/runs/*.json` and the test splits.

**Status 2026-10-31:** code fixes for B1-B9, B11-B14 and the A-row efficiency / K=3 items are in (see README "Issue fixes" and GitHub issues #1-#13); the rerun is `scripts/run_v2.sh`. Planner diagnosis: the CEM search is fine with oracle dynamics + oracle cost (90 % success); the student goal-latent cost is what fails (0 % even with oracle dynamics).

---

## A. Claim vs. evidence (P0)

| manuscript claim | what the repo shows | status |
|---|---|---|
| C2 student-only closed-loop navigation on unseen routes (Table 1) | latent CEM-MPC succeeds 3-7 % on nav, 0 % on arrange, at or below random (7 %); not run on NYC at all | **no evidence** |
| C1 three-term loss (feature alignment + prediction + policy imitation) | only the prediction term exists (`scripts/train.py`); no navigation head, no expert-imitation loss, no `L_feat` ablation | **not implemented** |
| C3 real vs. degree-preserving rewired vs. synthetic vs. GRU | exists, but controls are confounded (section B) and GRU wins on the synthetic arena | **partially** |
| geographically disjoint 70/15/15 split | one location (Times Square), random episode split, 120 train / 30 test | **not done** |
| blackout stress test 1/3/5 s during obstacle encounters, safe-stop metric | offline latent-MSE blackout only; no closed-loop blackout, no collisions, no SPL | **not done** |
| appearance / weather / density shifts | none | **not done** |
| efficiency: p50/p95 latency, FLOPs, memory on matched hardware | none; dense N×N matrix is materialised every step (`FlyBrain.W()`), so the sparse student is *not* cheap as implemented | **not done** |
| communicating connectomes K ∈ {1,2,4} with fixed-parameter control | K=2 gives 0.621 vs 0.629 (no gain, 1 seed); K=3 out of memory; no fixed-budget control | **null / incomplete** |
| humanoid transfer, FLYNN baseline | none | **not done** |

**Decision implied by the manuscript's own gate (A.1):** today this is a *representation-distillation pilot*, not an
autonomous-navigation result. Either (a) get E2 working on NYC with the full metric set, or (b) retitle and rescope
to "connectome-wired latent world models" and move navigation to future work. A spotlight with (b) needs the
confounds in B fixed and a much stronger head-direction story.

## B. Methodological confounds in the existing experiments

1. **(P0) "Degree-preserving shuffle" destroys region block structure, so "wiring identity" is not isolated.**
   `build_mask` permutes post-synaptic endpoints globally. Real CX→CX has 71,415 edges; the shuffle leaves 56,737
   and creates 9,741 CX→OUT and 5,771 VIS→CX edges (real: 237 and 619). The control therefore tests
   "real graph vs. graph with scrambled *macro-architecture*", not fine synaptic identity. Add a
   **block-preserving (stochastic-block / within-region degree-preserving) rewire**; if the real graph does not beat
   that, the claim collapses to "region-level architecture matters".
2. **(P1) Controls are not edge-matched.** Collapsing duplicate pairs leaves the shuffle with 90,388 unique edges
   (+32 self-loops) and the ER control with 91,595, vs. 91,663 real. Small, but the explainer says "same number
   of links". Use a proper switch-based rewiring (Maslov–Sneppen) that keeps exact counts and no self-loops.
3. **(P1) Sign handling differs across controls.** Shuffled keeps presynaptic signs (Dale's law intact); ER
   permutes signs per edge (Dale's law broken). Any shuffled-vs-random gap mixes topology with Dale's law.
4. **(P1) Signed vs. unsigned students use different parametrisations.** Signed: `softplus(W_raw)·sign`,
   init `|w|`; unsigned: raw `W_raw·0.8`. Different effective learning rates and initial spectral radius. The
   "signs do not help MSE" result is uninterpretable until both share a parametrisation (e.g. unsigned =
   signed with signs randomised but Dale-consistent).
5. **(P0) Only 1.3 % of the student's parameters are recurrent.** 91,663 edge weights vs. 5.33 M in `W_in`
   (of which 66 % feed non-VIS rows and are masked to zero, i.e. dead parameters) and 1.78 M in `W_out`.
   "7.25 M params" is misleading; the wiring comparison is a comparison of a ~92 k-parameter recurrence
   wrapped in 7 M of shared linear layers. Report recurrent / input / output params separately, drop the
   dead rows, and consider a smaller `W_in` so the recurrence has to do the work.
6. **(P1) GRU is not parameter-matched and not input-matched.** 9.50 M vs 7.25 M; the GRU also receives an
   explicit `visible` flag and obs at every unit, the FlyBrain receives obs only at VIS neurons and no flag.
   Dense uses a different lr (1e-3). No per-architecture hyperparameter search. A reviewer will attribute
   every gap to tuning.
7. **(P0) Teacher targets are non-causal.** Latents are computed per 8-step clip with bidirectional attention
   and each frame duplicated into one tubelet. The target at step t encodes frames up to the end of its clip,
   and the eval blind window (steps 8–15) is exactly clip 2. The student is asked to predict a latent that
   already contains the future it is blind to. Fix: causal/sliding-window teacher latents (clip ending at t),
   or state the leak and show it does not favour any wiring.
8. **(P0) No trivial baselines reported.** Recomputed on the same PCA-256 space:

   | | const-mean | copy-last-visible | best student |
   |---|---|---|---|
   | synthetic arena (blind h1–h8) | 1.043 | 0.97–1.19 | 0.544 (GRU) / 0.625 (connectome) |
   | NYC Times Square | 0.977 | 1.46–1.64 | 0.885 (connectome) |

   Students beat trivial baselines on the arena. On NYC the connectome is only **9 % below predicting the mean**
   (cosine 0.31), so the NYC headline is a weak effect from one seed on 30 episodes. Either scale NYC or present
   it as a pilot only.
9. **(P1) Heading "path integration" is mostly action counting.** Actions are discrete turns of fixed angle and
   are fed as one-hot input to CX/OUT neurons; relative heading = cumulative turn count. A leaky integrator
   decodes it trivially. Add (a) an **action-only linear baseline** (ridge on cumulative one-hot actions),
   (b) a probe-dimension-matched comparison (GRU has 1,400 features vs 6,935), (c) continuous or noisy
   turn angles so integration is non-trivial. The dense/GRU failure is still interesting but needs (a).
10. **(P1) EPG ring claim is negative.** 51 EPG neurons, R² ≤ 0.2. The plan's title claim ("emergent
    head-direction coding in EPG") must go, or be replaced by a population-level claim with the caveat in 9.
11. **(P1) Seeds and statistics.** 2 seeds for connectome/shuffled/random, 1 for GRU/dense/nosign/K=2, 1 for
    everything on NYC. No CIs, no paired tests across the 200 (or 30) shared test episodes. Minimum for a
    spotlight: 3–5 seeds everywhere, paired bootstrap CIs per horizon.
12. **(P2) Test blackout window is fixed at steps 8–15 while training windows are random.** Fine, but report
    at least one other window to show no position-specific overfitting.
13. **(P2) PCA-256 target** explains 94.6 % (arena) / 91.0 % (NYC) of teacher variance; MSE is reported in the
    whitened PCA space, which weights all components equally. Report MSE in the original 3,840-d space too.
14. **(P2) Planner (`eval_plan.py`)** encodes goals by rendering the goal with 8 headings and taking the
    nearest latent, CEM over 6 steps with 64 candidates, 2 iterations. It is an untuned planner; failing it
    says nothing. Before claiming C2, verify the planner with the *teacher* as the dynamics model (oracle
    upper bound) and with ground-truth state.

## C. Manuscript–code mismatches to reconcile

- Manuscript action space is unspecified; code uses 5 discrete actions, 24-step episodes, 110° camera at 256 px.
- Manuscript says FlyWire "selected subgraph"; code: FlyWire v783, 6,935 neurons (VIS 2,330, CX 2,875, MB 427,
  OUT 1,303), 91,663 edges with ≥5 synapses, sign from predicted neurotransmitter (ACh/DA/5-HT/OA +, Glu/GABA −),
  positive fraction 0.499. Put these in §3.3 and the licence (FlyWire CC-BY-NC) in §7.
- Teacher: `apiantonio/vjepa2.1-vit-base-384` HF conversion, ViT-B, 256 px crop, 16 frames (8 steps × 2),
  per-step target = mean token + 2×2 pooled tokens (3,840-d) → PCA-256. The manuscript says ViT variant
  "[VERIFY]"; it is ViT-B, not the ViT-L/g used in the V-JEPA 2.1 paper. State it as a limitation.
- The NYC twin is Valdrada (three.js/Electron, WebGL "medium" quality for training data); no weather, no
  pedestrians in the training renders (ambient traffic/pedestrians only in the demo renders). Manuscript
  §4.1 placeholders can be filled, but the 70/15/15 geographic split does not exist yet.
- Multi-connectome in code couples clones through contralateral edges moved to the partner brain, not
  "low-bandwidth learned messages" as the manuscript says. Pick one and describe it exactly.
- Videos/GIFs: "what the brain imagines" is a nearest-neighbour training frame to the predicted latent;
  label it as such in any figure caption. It is not a decoded image.
- References [5] FLYNN (arXiv 2607.00025), [6] (2602.17997), [7] FlyCNS (2609.28816) and [1] (2603.14482)
  could not be verified from this repo; confirm every arXiv id and author list before submission.
- LLM usage statement names ChatGPT only; this repo was built with Claude Code. Update the statement.

## D. Positioning risks (reviewer attacks to pre-empt)

- "Connectome as controller" is prior art (FLYNN, Lappalainen 2024, fly-locomotion graph models). The only
  defensible novelty is world-model distillation *into* the connectome + the matched-control analysis.
  That analysis is currently the weakest part (B1–B6).
- Reviewers from the connectomics side will ask why VIS/CX/MB/DN only, why 5-synapse threshold, and why
  visual input enters at TuBu/MeTu/LC rather than through the optic lobe. Justify or ablate the subgraph.
- CVPR audience will ask what this buys a robot. Without latency/FLOP measurements and closed-loop success
  there is no embodied-vision contribution; the paper reads as computational neuroscience.

## E. Suggested experiments, in priority order

1. Block-preserving rewire + Maslov–Sneppen rewire, Dale-consistent, 5 seeds, arena + NYC (fixes B1–B3).
2. Causal teacher targets (fixes B7); rerun everything.
3. Trivial baselines + action-only heading baseline in every table (B8, B9).
4. Shrink/match `W_in` so recurrent params are a meaningful fraction; GRU with equal total params and equal
   inputs; per-architecture lr sweep (B5, B6).
5. Planner validated with teacher-as-dynamics oracle, then student MPC on NYC with SPL/collision metrics (A).
6. Only if 5 works: geographic split, blackout-in-the-loop, appearance shifts, latency table.

---

## Paper-ready text: "Issues and Limitations" section

> **Limitations and open issues.** (i) *Scope of the controls.* Our degree-preserving rewire permutes
> post-synaptic targets globally and therefore also alters region-level block structure; it cannot separate the
> effect of synapse-level identity from that of macro-architecture. A block-preserving rewire is required
> before attributing gains to fine wiring. (ii) *Parameter accounting.* Recurrent synaptic weights constitute
> about 1.3 % of the student's trainable parameters; the input and readout layers, shared by all wirings,
> dominate the budget. The GRU baseline is not exactly parameter- or input-matched. (iii) *Teacher targets.*
> Per-step V-JEPA 2.1 latents are computed from 8-step clips with bidirectional attention, so targets inside
> the blind window encode frames the student does not see. (iv) *Trivial baselines.* On the NYC twin the best
> student improves on a constant-mean predictor by only 9 % (one seed, 30 test episodes); the NYC result is
> a pilot. (v) *Head-direction code.* Turn actions are discrete and fed to the recurrence, so relative heading
> is linearly decodable from integrated actions; the EPG sub-population alone does not encode heading
> (R² < 0.2), and we make no ring-attractor claim. (vi) *Closed-loop control.* Latent model-predictive control
> with the current planner does not exceed a random policy for any student, so no autonomous-navigation
> claim is made. (vii) *Statistics.* Several conditions have one or two seeds and no confidence intervals.
> (viii) *Biology.* Signs come from predicted neurotransmitters, dynamics are a leaky tanh rate model with a
> single time constant, and only four FlyWire populations (6,935 neurons, 91,663 edges with ≥5 synapses) are
> used; nothing here models neuronal physiology.
