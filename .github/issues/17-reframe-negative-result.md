---
title: "[P0] Reframe the paper around the negative prediction result"
labels: P0, writing
---
**Owner:** Amitava sir  ·  **Priority:** P0  ·  **Paper files:** `paper_v2/main.tex`, `sec/1_intro.tex`

## What the data says (run 2, 3 seeds, blind latent MSE at horizon 1 / 8, lower is better)
| core | h1 | h8 |
|---|---|---|
| connectome | 0.419 | 0.709 |
| block-shuffled | 0.426 | 0.711 |
| random (ER) | 0.459 | 0.690 |
| transformer | 0.402 | 0.611 |
| GRU | 0.348 | 0.525 |

The real wiring beats its block-shuffled control by 1.5 % at one step and is tied at eight steps, where random wiring is actually better. Parameter-matched free architectures are 15-25 % ahead.

## What is wrong with the text
The abstract and intro present the 1.5 % edge as the finding and never say that random wiring ties at h8. A reviewer who rebuilds Table 1 will reject the paper for framing.

## What to do
1. Lead with the negative: "synapse-level identity buys nothing on prediction; matched free architectures win".
2. Then the positive: heading and position are linearly decodable from the connectome state during blind windows, and blind homing works, where the GRU state does not carry this (numbers arrive from run 2 probes/homing).
3. Add the h8 row and the random-wiring tie to the scorecard; keep bootstrap CIs.
4. Reconsider the title's "what a real wiring diagram buys a latent predictor" unless the probes justify it.

## Done when
Abstract, intro and scorecard are rewritten and no claim in them is stronger than the tables.
