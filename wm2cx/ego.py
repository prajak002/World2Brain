"""Egomotion from video without labels: patch-grid phase correlation -> yaw (deg) and radial expansion (px).

Pure numpy. Used by scripts/ingest_video.py (E7 real-video benchmark); see that script for the conventions.
"""
import numpy as np


def gray(img): return img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


_HANN = {}
def hann(n):
    if n not in _HANN: w = np.hanning(n).astype(np.float32); _HANN[n] = np.outer(w, w)
    return _HANN[n]


def phase_corr(a_, b_):
    """Integer shift (dx, dy) that moves patch a onto patch b, plus peak sharpness (0..1)."""
    w = hann(a_.shape[0]); A = np.fft.fft2((a_ - a_.mean()) * w); B = np.fft.fft2((b_ - b_.mean()) * w)
    R = A * np.conj(B); R /= np.abs(R) + 1e-6; r = np.fft.ifft2(R).real
    iy, ix = np.unravel_index(np.argmax(r), r.shape); n = r.shape[0]
    dx = ix if ix <= n // 2 else ix - n; dy = iy if iy <= n // 2 else iy - n
    return -dx, -dy, float(r.max() / (np.abs(r).sum() / r.size + 1e-6) / r.size)


def egomotion(f0, f1, grid, fov_deg):
    """Patch-grid flow between two frames -> (yaw deg, expansion px). Yaw > 0 = scene shifts right = camera turned left."""
    g0, g1 = gray(f0), gray(f1); R = g0.shape[0]; P = R // grid; vecs, radial = [], []
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * P, gx * P
            dx, dy, q = phase_corr(g0[y0:y0 + P, x0:x0 + P], g1[y0:y0 + P, x0:x0 + P])
            if abs(dx) >= P // 2 or abs(dy) >= P // 2: continue           # wrapped / unreliable
            cx, cy = x0 + P / 2 - R / 2, y0 + P / 2 - R / 2; nr = np.hypot(cx, cy) + 1e-6
            vecs.append((dx, dy)); radial.append((dx * cx + dy * cy) / nr)
    if len(vecs) < grid:                                                   # nearly nothing tracked (dark / blur): no motion
        return 0.0, 0.0
    v = np.array(vecs, np.float32); yaw = float(np.median(v[:, 0])) * fov_deg / R
    return yaw, float(np.median(radial))


def quantise(yaw, exp_, turn_deg=6.0, fwd_px=1.0):
    """Arena action ids: 0 stay, 1 forward, 2 turn left (+yaw), 3 turn right, 4 back."""
    if abs(yaw) >= turn_deg: return 2 if yaw > 0 else 3
    if exp_ >= fwd_px: return 1
    if exp_ <= -fwd_px: return 4
    return 0
