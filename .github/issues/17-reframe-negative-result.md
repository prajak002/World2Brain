---
title: "[P0] Reframe the paper around the negative prediction result"
labels: P0, writing
---
**Owner:** Amitava sir

## Problem
Run 2 (3 seeds, blind MSE h1 / h8): connectome 0.419 / 0.709, block-shuffled 0.426 / 0.711, random 0.459 / 0.690, transformer 0.402 / 0.611, GRU 0.348 / 0.525. The abstract and intro present the 1.5 % edge over block-shuffled as the finding and do not state that random wiring is better at h8. A reviewer rebuilding Table 1 will reject on framing alone.

## Fix
1. Abstract and §1: lead with "synapse-level identity buys nothing on prediction; matched free architectures win by 15-25 %", then the state finding.
2. Add the h8 row and the random-wiring tie to the scorecard in §4; keep bootstrap CIs.
3. Title: drop "what a real wiring diagram buys a latent predictor" unless the probes justify it.
