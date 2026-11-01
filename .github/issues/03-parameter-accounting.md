---
title: "[P0] Only 1.3% of student parameters are recurrent; W_in has dead rows"
labels: P0, model
---
## Problem
Breakdown for K=1 (7.25M params): `W_in` 5.33M, `W_out` 1.78M, recurrent edges 91,663 (1.3%). 66% of `W_in` rows feed non-VIS neurons and are masked to zero every step: dead parameters that still count in the "parameter-matched" comparison.

## Fix
- Make `W_in` map only to VIS neurons (2,330 rows).
- Report recurrent / input / output params separately in every table.
- Consider a low-rank or smaller `W_in` so the recurrence has to carry the prediction.
