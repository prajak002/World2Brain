---
title: "[P2] Larger teacher and real odometry for the real-video stage"
labels: P2, data
---
**Owner:** Priyanshi

## Problem
Single ViT-B teacher at 256 px; real-video actions are flow-estimated, so the heading probe on real clips has no ground truth.

## Fix
ViT-L at 384 px (one GPU day, re-collect targets). Add one public egocentric set with odometry (RECON or SCAND) to `ingest_video.py` so flow-estimated actions can be validated.
