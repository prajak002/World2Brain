"""NYC digital-twin arena: real Midtown Manhattan building footprints + roof heights (NYC Open Data 5zhs-2jue)
and the OpenStreetMap street network, rendered as an egocentric 360-degree cylindrical panorama by raycasting
against the real building geometry. Facades get procedural windows/materials keyed to the real construction year
and height; sun shading uses the wall normal; distance fog; road/sidewalk ground.
Actions: 0 stay, 1 forward 3 m, 2 turn left 22.5 deg, 3 turn right, 4 back 2 m.
Task: navigate to a goal street location (success < 12 m). Shortest paths come from a 2 m walkability grid.
"""
from __future__ import annotations
import json, numpy as np
from collections import deque

FT = 0.3048


class NYCArena:
    def __init__(self, buildings="data/nyc/midtown_buildings.json", streets="data/nyc/midtown_streets.json",
                 res=256, seed=0, radius=160.0, episode_goal_dist=(40.0, 90.0)):
        self.res = res; self.rng = np.random.default_rng(seed); self.radius = radius; self.gd = episode_goal_dist
        B = json.load(open(buildings)); S = json.load(open(streets))
        lat0 = np.mean([float(c[1]) for b in B for c in b["the_geom"]["coordinates"][0][0]]); lon0 = np.mean([float(c[0]) for b in B for c in b["the_geom"]["coordinates"][0][0]])
        self.lat0, self.lon0 = lat0, lon0
        self.m_per_deg = (111320.0 * np.cos(np.radians(lat0)), 110540.0)
        polys, heights, years, ids = [], [], [], []
        for b in B:
            for poly in b["the_geom"]["coordinates"]:
                ring = np.array([self.xy(float(c[0]), float(c[1])) for c in poly[0]])
                h = float(b.get("height_roof", 10) or 10) * FT
                polys.append(ring); heights.append(max(h, 3.0)); years.append(int(float(b.get("construction_year", 1950) or 1950))); ids.append(int(b["bin"]))
        self.polys, self.heights, self.years, self.bins = polys, np.array(heights), np.array(years), np.array(ids)
        # edge list for raycasting
        A, Bp, pid = [], [], []
        for i, ring in enumerate(polys):
            A.append(ring[:-1]); Bp.append(ring[1:]); pid.append(np.full(len(ring) - 1, i))
        self.EA, self.EB, self.Eid = np.concatenate(A), np.concatenate(Bp), np.concatenate(pid)
        self.Ed = self.EB - self.EA; self.Elen = np.linalg.norm(self.Ed, axis=1) + 1e-9
        self.Enorm = np.stack([self.Ed[:, 1], -self.Ed[:, 0]], 1) / self.Elen[:, None]
        # street nodes
        nodes = {e["id"]: self.xy(e["lon"], e["lat"]) for e in S["elements"] if e["type"] == "node"}
        self.street_pts = np.array([nodes[n] for e in S["elements"] if e["type"] == "way" for n in e["nodes"] if n in nodes])
        self.street_segs = [(nodes[w["nodes"][i]], nodes[w["nodes"][i + 1]]) for w in S["elements"] if w["type"] == "way" for i in range(len(w["nodes"]) - 1) if w["nodes"][i] in nodes and w["nodes"][i + 1] in nodes]
        # walkability grid (2 m): outside buildings
        allpts = np.concatenate(polys); self.xmin, self.ymin = allpts.min(0) - 20; self.xmax, self.ymax = allpts.max(0) + 20
        self.cell = 2.0; W = int((self.xmax - self.xmin) / self.cell) + 1; H = int((self.ymax - self.ymin) / self.cell) + 1
        self.walk = np.ones((H, W), bool)
        for ring, h in zip(polys, heights):
            x0, y0 = np.floor((ring.min(0) - [self.xmin, self.ymin]) / self.cell).astype(int); x1, y1 = np.ceil((ring.max(0) - [self.xmin, self.ymin]) / self.cell).astype(int)
            xs = self.xmin + np.arange(max(x0, 0), min(x1 + 1, W)) * self.cell; ys = self.ymin + np.arange(max(y0, 0), min(y1 + 1, H)) * self.cell
            gx, gy = np.meshgrid(xs, ys); inside = self._inside(ring, np.stack([gx.ravel(), gy.ravel()], 1)).reshape(gy.shape)
            self.walk[max(y0, 0):max(y0, 0) + inside.shape[0], max(x0, 0):max(x0, 0) + inside.shape[1]] &= ~inside
        self.cam_h = 1.7
        self.reset()

    def xy(self, lon, lat):
        return np.array([(lon - self.lon0) * self.m_per_deg[0], (lat - self.lat0) * self.m_per_deg[1]])

    @staticmethod
    def _inside(ring, pts):
        x, y = pts[:, 0], pts[:, 1]; inside = np.zeros(len(pts), bool); n = len(ring) - 1
        for i in range(n):
            x1, y1 = ring[i]; x2, y2 = ring[i + 1]
            c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1); inside ^= c
        return inside

    def _grid(self, p): return int((p[1] - self.ymin) / self.cell), int((p[0] - self.xmin) / self.cell)
    def walkable(self, p):
        r, c = self._grid(p); return 0 <= r < self.walk.shape[0] and 0 <= c < self.walk.shape[1] and self.walk[r, c]

    def bfs_next(self, start, goal):
        """BFS on the walkability grid from goal; returns next waypoint direction from start."""
        if not hasattr(self, "_dist") or self._dist_goal != tuple(goal):
            H, W = self.walk.shape; dist = np.full((H, W), -1, np.int32); gr, gc = self._grid(goal); dist[gr, gc] = 0; q = deque([(gr, gc)])
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and self.walk[rr, cc] and dist[rr, cc] < 0: dist[rr, cc] = dist[r, c] + 1; q.append((rr, cc))
            self._dist, self._dist_goal = dist, tuple(goal)
        r, c = self._grid(start); best, bd = None, self._dist[r, c] if self._dist[r, c] >= 0 else 1e9
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.walk.shape[0] and 0 <= cc < self.walk.shape[1] and 0 <= self._dist[rr, cc] < bd: best, bd = (rr, cc), self._dist[rr, cc]
        if best is None: return goal - start
        # follow the descending-distance path a few cells further to get a smoother waypoint
        r, c = best
        for _ in range(3):
            nb = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= r + dr < self.walk.shape[0] and 0 <= c + dc < self.walk.shape[1] and self._dist[r + dr, c + dc] >= 0]
            if not nb: break
            r, c = min(nb, key=lambda rc: self._dist[rc])
        return np.array([self.xmin + c * self.cell, self.ymin + r * self.cell]) - start

    # ---------------- episode ----------------
    def reset(self):
        for _ in range(200):
            p = self.street_pts[self.rng.integers(len(self.street_pts))] + self.rng.normal(0, 2, 2)
            if self.walkable(p) and np.linalg.norm(p) < self.radius * 3: break
        self.pos = p; self.heading = self.rng.uniform(-np.pi, np.pi)
        dd = np.linalg.norm(self.street_pts - p, axis=1)
        cand = np.where((dd > self.gd[0]) & (dd < self.gd[1]))[0]
        cand = [i for i in cand if self.walkable(self.street_pts[i])]
        if not cand: return self.reset()
        self.goal = self.street_pts[self.rng.choice(cand)]; self.t = 0
        if hasattr(self, "_dist"): del self._dist
        return self.render()

    def step(self, a):
        d, dth = {0: (0, 0), 1: (3.0, 0), 2: (0, np.pi / 8), 3: (0, -np.pi / 8), 4: (-2.0, 0)}[int(a)]
        self.heading = (self.heading + dth + np.pi) % (2 * np.pi) - np.pi
        if d:
            new = self.pos + d * np.array([np.cos(self.heading), np.sin(self.heading)])
            if self.walkable(new): self.pos = new
        self.t += 1
        return self.render(), self.success()

    def success(self): return float(np.linalg.norm(self.pos - self.goal) < 12.0)
    def goal_distance(self): return float(np.linalg.norm(self.pos - self.goal))

    def policy(self, rng, eps=0.2):
        if rng.random() < eps: return int(rng.integers(1, 4))
        v = self.bfs_next(self.pos, self.goal)
        rel = (np.arctan2(v[1], v[0]) - self.heading + np.pi) % (2 * np.pi) - np.pi
        if abs(rel) > np.pi / 8: return 2 if rel > 0 else 3
        ahead = self.pos + 3.0 * np.array([np.cos(self.heading), np.sin(self.heading)])
        if not self.walkable(ahead): return 2 if rel >= 0 else 3
        return 1

    # ---------------- rendering ----------------
    def _cast(self, pos, heading):
        R = self.res; ang = heading + np.linspace(np.pi, -np.pi, R, endpoint=False); D = np.stack([np.cos(ang), np.sin(ang)], 1)
        near = np.linalg.norm((self.EA + self.EB) / 2 - pos, axis=1) < self.radius
        A, Ed, El, En, Eid = self.EA[near], self.Ed[near], self.Elen[near], self.Enorm[near], self.Eid[near]
        # ray p + t D ; segment A + u Ed, u in [0,1]
        den = D[:, None, 0] * (-Ed[None, :, 1]) - D[:, None, 1] * (-Ed[None, :, 0])  # [R,E]
        ap = A[None] - pos
        t = (ap[..., 0] * (-Ed[None, :, 1]) - ap[..., 1] * (-Ed[None, :, 0])) / np.where(np.abs(den) < 1e-9, np.inf, den)
        u = (D[:, None, 0] * ap[..., 1] - D[:, None, 1] * ap[..., 0]) / np.where(np.abs(den) < 1e-9, np.inf, den)
        ok = (t > 0.3) & (u >= 0) & (u <= 1); t = np.where(ok, t, np.inf)
        j = t.argmin(1); tmin = t[np.arange(R), j]; hit = np.isfinite(tmin)
        return tmin, hit, Eid[j], u[np.arange(R), j] * El[j], En[j], D

    def render(self, pos=None, heading=None):
        pos = self.pos if pos is None else pos; heading = self.heading if heading is None else heading
        R = self.res; horizon = R // 2; f = R / (2 * np.pi)
        img = np.zeros((R, R, 3), np.float32)
        # sky with sun glow; ground: asphalt + sidewalk band
        sky = np.linspace([95, 140, 215], [200, 220, 240], horizon)[:, None, :]
        img[:horizon] = sky
        rows = np.arange(horizon, R); ground_t = f * self.cam_h / np.maximum(rows - horizon, 1)
        img[horizon:] = np.linspace([60, 60, 62], [45, 45, 47], R - horizon)[:, None, :]
        tmin, hit, pid, along, nrm, D = self._cast(pos, heading)
        sun = np.array([np.cos(np.radians(230)), np.sin(np.radians(230))])
        fog = lambda d: np.clip(d / 420.0, 0, 0.85)[:, None] if np.ndim(d) else np.clip(d / 420.0, 0, 0.85)
        for c in range(R):
            if not hit[c]: continue
            d = tmin[c]; i = pid[c]; h = self.heights[i]; yr = self.years[i]
            top = int(np.clip(f * (h - self.cam_h) / d, 0, horizon)); bot = int(np.clip(f * self.cam_h / d, 0, R - horizon))
            # material from real attributes: modern tall -> glass; pre-war -> brick/limestone
            if yr >= 1960 or h > 120: base = np.array([120, 150, 175]) if (i % 3) else np.array([90, 100, 110]); glass = True
            elif yr >= 1930: base = np.array([170, 150, 120]); glass = False
            else: base = np.array([150, 80, 60]) if (i % 2) else np.array([190, 170, 140]); glass = False
            shade = 0.55 + 0.45 * max(0.0, float(nrm[c] @ sun)) if np.ndim(nrm[c]) else 0.8
            ys = np.arange(horizon - top, horizon)
            zworld = (horizon - ys) * d / f + self.cam_h  # metres above ground per pixel row
            floor_u = (zworld % 3.6) / 3.6; win_u = (along[c] % 4.0) / 4.0
            win = ((floor_u > 0.25) & (floor_u < 0.8) & (win_u > 0.15) & (win_u < 0.85))
            col = np.tile(base * shade, (len(ys), 1))
            if glass: col[win] = col[win] * 0.6 + np.array([40, 60, 90]) * 0.4 + 30 * np.sin(zworld[win] / 7.0)[:, None]
            else: col[win] = col[win] * 0.35
            col[(zworld % 3.6) < 0.25] *= 0.85  # floor slabs
            col = col * (1 - fog(d)) + np.array([190, 200, 215]) * fog(d)
            img[ys, c] = col
            # sidewalk band at the building foot
            sb = np.arange(horizon, min(horizon + bot, R)); sw = (ground_t[: len(sb)] > d * 0.75) if len(sb) else np.zeros(0, bool)
            img[sb[sw], c] = np.array([150, 148, 140]) * (1 - fog(d)) + np.array([190, 200, 215]) * fog(d)
        # road markings: dashed centre lines where the ray hits a street segment are omitted for cost; add lane texture
        return np.clip(img, 0, 255).astype(np.uint8)

    def topdown(self, res=256, extent=220.0):
        img = np.full((res, res, 3), 235, np.uint8); s = res / (2 * extent); o = self.pos
        for ring, h in zip(self.polys, self.heights):
            if np.linalg.norm(ring.mean(0) - o) > extent * 1.5: continue
            pts = ((ring - o) * s + res / 2).astype(int)
            for (x, y) in pts:
                if 0 <= x < res and 0 <= y < res: img[y, x] = (90, 90, 120) if h > 60 else (160, 140, 130)
        g = ((self.goal - o) * s + res / 2).astype(int); img[np.clip(g[1] - 3, 0, res - 1):g[1] + 3, np.clip(g[0] - 3, 0, res - 1):g[0] + 3] = (220, 40, 40)
        img[res // 2 - 3:res // 2 + 3, res // 2 - 3:res // 2 + 3] = 0
        return img[::-1]
