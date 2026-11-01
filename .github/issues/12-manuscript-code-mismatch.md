---
title: "[P2] Reconcile manuscript text with implementation"
labels: P2, writing
---
## Mismatches
- Loss: manuscript has feature alignment + prediction + policy imitation; code has prediction only. No `L_feat`/`L_pred` ablations exist.
- Action space unspecified in manuscript; code: 5 discrete actions, 24-step episodes, 110° camera at 256 px.
- Teacher: `apiantonio/vjepa2.1-vit-base-384`, **ViT-B**, 16 frames, target = mean token + 2×2 pooled (3,840-d) → PCA-256 (94.6% / 91.0% variance).
- Connectome: FlyWire v783, 6,935 neurons (VIS 2,330, CX 2,875, MB 427, OUT 1,303), 91,663 edges ≥5 synapses, signs from predicted NT, 49.9% excitatory. Licence CC-BY-NC.
- Multi-brain: code moves contralateral edges to the partner clone; manuscript says "low-bandwidth learned messages".
- NYC twin is Valdrada (three.js/Electron); state render quality used for training.
- "What the brain imagines" frames are nearest-neighbour training frames, not decoded images; caption accordingly.
- Verify arXiv ids for refs [1], [5], [6], [7].
- LLM usage statement names ChatGPT only; this repo was built with Claude Code.
- Explainer says "same number of links"; not exactly true (see #1).
