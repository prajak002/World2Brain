"""NYC digital-twin arena on Valdrada (https://github.com/mkturkcan/valdrada): real NYC building footprints, PLUTO
attributes, street centrelines and street furniture, rendered by the Valdrada server and driven over its CARLA-style API.

Same discrete interface as wm2cx.env.FlyArena / wm2cx.nyc.NYCArena:
  actions 0 stay, 1 forward 4 m, 2 turn left 22.5 deg, 3 turn right 22.5 deg, 4 back 2 m
  task: walk to a goal street location 40-90 m away (success < 12 m); scripted policy follows the sidewalk route.
The agent is the spectator (teleported each step; the city streams around it) with a forward RGB camera at eye height.
"""
from __future__ import annotations
import math, time, numpy as np
import valdrada
from valdrada import Location, Rotation, Transform

WALKABLE = {"sidewalk", "curb", "grass", "path", "brick", "warn", "warnIron", "asphalt", "paint", "gutter", "busred", "crosswalk", "plaza", "warnIron"}


class ValdradaArena:
    def __init__(self, host="127.0.0.1", port=2000, lat=40.7580, lon=-73.9855, radius=140.0, seed=0, fov=110,
                 episode_goal_dist=(30.0, 60.0), settle_ticks=1, ambient=(0, 0)):
        self.rng = np.random.default_rng(seed); self.gd = episode_goal_dist; self.radius = radius; self.settle = settle_ticks
        self.client = valdrada.Client(host, port); self.client.set_timeout(300.0)
        self.world = self.client.get_world(); self.m = self.world.get_map()
        self.world.apply_settings(valdrada.WorldSettings(synchronous_mode=True, fixed_delta_seconds=0.05))
        try: self.world.set_ambient_traffic(vehicles=ambient[0], walkers=ambient[1])
        except Exception as e: print("ambient traffic:", e)
        self.centre = self.m.geolocation_to_location(lat, lon)
        self.spec = self.world.get_spectator()
        self.spec.set_transform(Transform(self.centre + Location(0, 0, 1.7)))
        print("valdrada loaded:", self.world.wait_until_loaded())
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb"); bp.set_attribute("fov", fov)
        self.cam = self.world.spawn_actor(bp, Transform(Location(0, 0, 0)), attach_to=self.spec)
        self._img = {}; self.cam.listen(lambda im: self._img.update(im=im))
        # candidate street points: lane spawn points pushed to the nearest sidewalk-ish spot
        sp = self.m.get_spawn_points(center=self.centre, radius=radius, spacing=15)
        self.street_pts = np.array([[t.location.x, t.location.y] for t in sp])
        print(f"{len(self.street_pts)} street points within {radius} m")
        self.cam_h = 1.7; self.t = 0

    # ---------------- geometry helpers ----------------
    def surface(self, p):
        try: return self.m.get_surface(Location(float(p[0]), float(p[1]), 0.0))
        except Exception: return None

    def near_route(self, p, tol=4.0):
        """distance from p to the planned sidewalk/crosswalk polyline (route points are walkable by construction)"""
        if not hasattr(self, "path") or len(self.path) < 2: return False
        A, B = self.path[:-1], self.path[1:]; AB = B - A; L2 = (AB ** 2).sum(1) + 1e-9
        u = np.clip(((p - A) * AB).sum(1) / L2, 0, 1); proj = A + u[:, None] * AB
        return float(np.linalg.norm(proj - p, axis=1).min()) < tol

    def walkable(self, p):
        if self.near_route(p): return True
        s = self.surface(p)
        return bool(s) and s.get("height") is not None and (s.get("kind") in WALKABLE)

    def ground(self, p):
        s = self.surface(p); return float(s["height"]) if s and s.get("height") is not None else 0.0

    # ---------------- episode ----------------
    def reset(self):
        self.path = np.zeros((0, 2))
        for attempt in range(400):
            slack = 1.5 if attempt < 100 else (2.0 if attempt < 250 else 4.0)   # loosen the route-directness requirement if needed
            p = self.street_pts[self.rng.integers(len(self.street_pts))] + self.rng.normal(0, 1.5, 2)
            if not self.walkable(p): continue
            dd = np.linalg.norm(self.street_pts - p, axis=1); cand = np.where((dd > self.gd[0]) & (dd < self.gd[1]))[0]
            self.rng.shuffle(cand)
            for i in cand[:8]:
                g = self.street_pts[i]
                try: length, path = self.m.plan_walk_route(Location(*map(float, p), 0.0), Location(*map(float, g), 0.0))
                except Exception: continue
                if length < slack * np.linalg.norm(g - p) + 10:
                    self.pos, self.goal, self.path = np.array(p, float), np.array(g, float), np.array([[q.x, q.y] for q in path])
                    self.heading = float(self.rng.uniform(-np.pi, np.pi)); self.t = 0; self.wp = 0
                    return self.render()
        raise RuntimeError("no start/goal pair found")

    def step(self, a):
        d, dth = {0: (0, 0), 1: (4.0, 0), 2: (0, np.pi / 8), 3: (0, -np.pi / 8), 4: (-2.0, 0)}[int(a)]
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
        # monotonic route index: advance when within 4 m of the current waypoint; replan when far off the route
        while self.wp < len(self.path) - 1 and np.linalg.norm(self.path[self.wp] - self.pos) < 4.0: self.wp += 1
        if np.linalg.norm(self.path - self.pos, axis=1).min() > 12:
            try:
                _, path = self.m.plan_walk_route(Location(*map(float, self.pos), 0.0), Location(*map(float, self.goal), 0.0))
                self.path = np.array([[q.x, q.y] for q in path]); self.wp = 1 if len(self.path) > 1 else 0
            except Exception: pass
        v = self.path[self.wp] - self.pos
        rel = (np.arctan2(v[1], v[0]) - self.heading + np.pi) % (2 * np.pi) - np.pi
        if abs(rel) > np.pi / 8: return 2 if rel > 0 else 3
        ahead = self.pos + 4.0 * np.array([np.cos(self.heading), np.sin(self.heading)])
        if not self.walkable(ahead): return 2 if rel >= 0 else 3
        return 1

    # ---------------- rendering ----------------
    def render(self, pos=None, heading=None):
        pos = self.pos if pos is None else pos; heading = self.heading if heading is None else heading
        z = self.ground(pos) + self.cam_h
        self.spec.set_transform(Transform(Location(float(pos[0]), float(pos[1]), z), Rotation(pitch=0, yaw=math.degrees(heading))))
        for _ in range(self.settle): self.world.tick()
        im = self._img["im"].to_numpy()[..., :3]
        return np.ascontiguousarray(im)

    def close(self):
        try: self.cam.stop(); self.cam.destroy()
        except Exception: pass
