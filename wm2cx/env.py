"""FlyArena: egocentric 360-degree panoramic arena with landmarks and pushable boxes.

A fly-like agent (compound eye = full horizontal panorama) moves in a square arena.
Objects are vertical cylinders/boxes; render is a cheap column-wise raycast.
Tasks:
  nav     : reach the goal landmark (identified by colour)
  arrange : push coloured boxes into matching coloured target zones
Actions (discrete): 0 stay, 1 forward, 2 turn left, 3 turn right, 4 back
"""
from __future__ import annotations
import numpy as np

ACTIONS = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, +np.pi / 8), 3: (0.0, -np.pi / 8), 4: (-0.6, 0.0)}
COLORS = np.array([[220, 60, 60], [60, 200, 80], [70, 90, 230], [235, 200, 50], [200, 70, 200], [60, 210, 210]], np.uint8)


class FlyArena:
    def __init__(self, task="nav", size=12.0, n_landmarks=4, n_boxes=2, res=256, seed=0, turn_noise=0.0):
        """turn_noise > 0 makes every turn a continuous angle, pi/8 * (1 + N(0, turn_noise)), so relative heading is no
        longer a linear function of the discrete action counts and path integration becomes non-trivial (#7)."""
        self.task, self.size, self.res = task, size, res
        self.nl, self.nb = n_landmarks, n_boxes if task == "arrange" else 0
        self.rng = np.random.default_rng(seed); self.turn_noise = turn_noise
        self.step_len, self.fov_cols = 0.6, res
        self.reset()

    # ---------------- state ----------------
    def reset(self):
        r = self.rng
        self.lm = r.uniform(1.0, self.size - 1.0, (self.nl, 2))  # landmark positions (static, tall)
        self.lm_col = r.permutation(len(COLORS))[: self.nl]
        self.goal = int(r.integers(self.nl))
        self.box = r.uniform(3.0, self.size - 3.0, (self.nb, 2))
        self.box_col = r.permutation(len(COLORS))[: self.nb]
        self.zone = r.uniform(3.0, self.size - 3.0, (self.nb, 2))
        for i in range(self.nb):  # targets start well away from their boxes
            while np.linalg.norm(self.zone[i] - self.box[i]) < 2.5:
                self.zone[i] = r.uniform(3.0, self.size - 3.0, 2)
        self.pos = r.uniform(1.5, self.size - 1.5, 2)
        self.heading = r.uniform(-np.pi, np.pi)
        self.t = 0
        return self.render()

    def state(self):
        return dict(pos=self.pos.copy(), heading=float(self.heading), box=self.box.copy(), t=self.t)

    def set_state(self, s):
        self.pos, self.heading, self.box, self.t = s["pos"].copy(), float(s["heading"]), s["box"].copy(), s.get("t", self.t)

    def step(self, a: int, render=True):
        """Apply action a. render=False skips the panorama (returns None) for ground-truth-state planning rollouts."""
        d, dth = ACTIONS[int(a)]
        if dth != 0.0 and self.turn_noise > 0: dth *= 1.0 + self.turn_noise * self.rng.normal()
        self.heading = (self.heading + dth + np.pi) % (2 * np.pi) - np.pi
        if d != 0.0:
            new = self.pos + d * self.step_len * np.array([np.cos(self.heading), np.sin(self.heading)])
            new = np.clip(new, 0.5, self.size - 0.5)
            # push boxes
            mv = d * self.step_len * np.array([np.cos(self.heading), np.sin(self.heading)])
            for i in range(self.nb):
                v = self.box[i] - new
                if np.linalg.norm(v) < 0.8 and np.dot(v, mv) > 0:  # only push when moving into the box
                    self.box[i] = np.clip(self.box[i] + mv, 1.5, self.size - 1.5)
            self.pos = new  # landmarks are visual beacons, not obstacles
        self.t += 1
        return (self.render() if render else None), self.success()

    def success(self):
        if self.task == "nav":
            return float(np.linalg.norm(self.pos - self.lm[self.goal]) < 1.2)
        return float(np.all(np.linalg.norm(self.box - self.zone, axis=1) < 1.0))

    def goal_distance(self):
        if self.task == "nav":
            return float(np.linalg.norm(self.pos - self.lm[self.goal]))
        return float(np.linalg.norm(self.box - self.zone, axis=1).mean())

    # ---------------- rendering ----------------
    def _cylinders(self):
        objs = []
        for i in range(self.nl):
            objs.append((self.lm[i], 0.5, 2.5, COLORS[self.lm_col[i]]))
        for i in range(self.nb):
            objs.append((self.box[i], 0.45, 0.9, COLORS[self.box_col[i]]))
            objs.append((self.zone[i], 0.9, 0.08, (COLORS[self.box_col[i]] * 0.6).astype(np.uint8)))
        return objs

    def render(self, pos=None, heading=None):
        pos = self.pos if pos is None else pos
        heading = self.heading if heading is None else heading
        R = self.res
        img = np.zeros((R, R, 3), np.uint8)
        horizon = R // 2
        # sky / floor gradient
        img[:horizon] = np.linspace([150, 180, 230], [200, 215, 240], horizon, dtype=np.uint8)[:, None, :]
        img[horizon:] = np.linspace([120, 110, 95], [70, 60, 50], R - horizon, dtype=np.uint8)[:, None, :]
        # arena walls: 4 walls with slightly different shades to disambiguate
        angles = heading + np.linspace(np.pi, -np.pi, R, endpoint=False)
        dirs = np.stack([np.cos(angles), np.sin(angles)], 1)
        wall_t = np.full(R, np.inf)
        wall_shade = np.zeros(R, np.uint8)
        for k, (n, c) in enumerate([((1, 0), 0.0), ((-1, 0), self.size), ((0, 1), 0.0), ((0, -1), self.size)]):
            n = np.array(n, float)
            denom = dirs @ n
            tt = np.where(np.abs(denom) > 1e-6, (c * (-n).sum() if False else 0), 0)
            # plane: n.x = -c  (x=0 -> n=(1,0),c=0 ; x=size -> n=(-1,0), c=-size handled below)
            if k == 0:  tt = (0 - pos[0]) / np.where(np.abs(dirs[:, 0]) > 1e-6, dirs[:, 0], 1e-6)
            if k == 1:  tt = (self.size - pos[0]) / np.where(np.abs(dirs[:, 0]) > 1e-6, dirs[:, 0], 1e-6)
            if k == 2:  tt = (0 - pos[1]) / np.where(np.abs(dirs[:, 1]) > 1e-6, dirs[:, 1], 1e-6)
            if k == 3:  tt = (self.size - pos[1]) / np.where(np.abs(dirs[:, 1]) > 1e-6, dirs[:, 1], 1e-6)
            ok = (tt > 0) & (tt < wall_t)
            wall_t[ok] = tt[ok]; wall_shade[ok] = 90 + 25 * k
        wall_h = np.clip((R * 0.25 / np.maximum(wall_t, 1e-3)).astype(int), 0, horizon)
        depth = wall_t.copy()
        for c in range(R):
            h = wall_h[c]
            img[horizon - h:horizon + h // 2, c] = (wall_shade[c], wall_shade[c], wall_shade[c] + 20)
        # objects, far to near
        objs = self._cylinders()
        objs.sort(key=lambda o: -np.linalg.norm(o[0] - pos))
        for (p, rad, hgt, col) in objs:
            v = p - pos
            dist = np.linalg.norm(v)
            if dist < 1e-3: continue
            ang = np.arctan2(v[1], v[0])
            half = np.arcsin(min(rad / dist, 1.0))
            rel = (ang - heading + np.pi) % (2 * np.pi) - np.pi
            c0 = int((np.pi - rel - half) / (2 * np.pi) * R); c1 = int((np.pi - rel + half) / (2 * np.pi) * R)
            top = int(np.clip(R * 0.25 * hgt / dist, 0, horizon)); bot = int(np.clip(R * 0.25 * 0.6 / dist, 0, horizon))
            shade = np.clip(1.0 - dist / (1.6 * self.size), 0.35, 1.0)
            for c in range(c0, c1 + 1):
                cc = c % R
                if dist < depth[cc]:
                    img[horizon - top:horizon + bot, cc] = (col * shade).astype(np.uint8)
        return img

    def topdown(self, res=256):
        img = np.full((res, res, 3), 240, np.uint8)
        s = res / self.size
        for (p, rad, hgt, col) in self._cylinders():
            cx, cy = int(p[0] * s), int(p[1] * s); r = max(int(rad * s), 2)
            yy, xx = np.ogrid[:res, :res]; m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            img[m] = col
        cx, cy = int(self.pos[0] * s), int(self.pos[1] * s)
        img[max(cy - 3, 0):cy + 3, max(cx - 3, 0):cx + 3] = 0
        hx, hy = int(cx + 10 * np.cos(self.heading)), int(cy + 10 * np.sin(self.heading))
        img[np.clip(hy, 0, res - 1), np.clip(hx, 0, res - 1)] = 0
        return img


def scripted_policy(env: FlyArena, rng, eps=0.3):
    """Noisy goal-directed policy used to collect training video."""
    if env.task == "nav":
        target = env.lm[env.goal]
    else:
        i = int(np.argmax(np.linalg.norm(env.box - env.zone, axis=1)))
        b, z = env.box[i], env.zone[i]
        u = (z - b) / (np.linalg.norm(z - b) + 1e-6)
        push_from = np.clip(b - 1.3 * u, 0.6, env.size - 0.6)
        ab = b - env.pos
        aligned = np.dot(ab / (np.linalg.norm(ab) + 1e-6), u) > 0.9
        target = z if (np.linalg.norm(ab) < 1.6 and aligned) else push_from
    if rng.random() < eps:
        return int(rng.integers(1, 5))
    v = target - env.pos
    rel = (np.arctan2(v[1], v[0]) - env.heading + np.pi) % (2 * np.pi) - np.pi
    if abs(rel) > np.pi / 8: return 2 if rel > 0 else 3
    return 1
