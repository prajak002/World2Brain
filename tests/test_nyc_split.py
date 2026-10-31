"""NYC arena (appearance shifts, state round trip, collisions) on a tiny synthetic city, and the geographic split tool."""
import json, subprocess, sys, os, numpy as np, pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny_city(tmp_path):
    """3x3 grid of square buildings (40 m blocks, 20 m streets) around lat/lon 0; streets along the gaps."""
    lat0, lon0 = 40.0, -74.0; mx, my = 111320.0 * np.cos(np.radians(lat0)), 110540.0
    def ll(x, y): return [lon0 + x / mx, lat0 + y / my]
    B = []
    for i in range(3):
        for j in range(3):
            x0, y0 = -90 + i * 60, -90 + j * 60
            ring = [ll(x0, y0), ll(x0 + 40, y0), ll(x0 + 40, y0 + 40), ll(x0, y0 + 40), ll(x0, y0)]
            B.append(dict(bin=i * 3 + j, height_roof=100 + 50 * i, construction_year=1920 + 30 * j, the_geom=dict(coordinates=[[ring]])))
    nodes, ways, nid = [], [], 0
    for k in range(4):                       # streets in the gaps between building columns / rows
        c = -110 + k * 60 + 10 if k else -110
        xs = [c] * 9; ys = list(np.linspace(-110, 110, 9))
        ids = []
        for x, y in zip(xs, ys): nodes.append(dict(type="node", id=nid, lon=ll(x, y)[0], lat=ll(x, y)[1])); ids.append(nid); nid += 1
        ways.append(dict(type="way", id=1000 + k, nodes=ids))
        ids = []
        for x, y in zip(ys, xs): nodes.append(dict(type="node", id=nid, lon=ll(x, y)[0], lat=ll(x, y)[1])); ids.append(nid); nid += 1
        ways.append(dict(type="way", id=2000 + k, nodes=ids))
    bf, sf = tmp_path / "b.json", tmp_path / "s.json"
    bf.write_text(json.dumps(B)); sf.write_text(json.dumps(dict(elements=nodes + ways)))
    return str(bf), str(sf)


def test_nyc_arena_appearance_state_and_collisions(tmp_path):
    from wm2cx.nyc import NYCArena
    bf, sf = tiny_city(tmp_path)
    env = NYCArena(bf, sf, res=64, seed=0, radius=200.0, episode_goal_dist=(30.0, 120.0)); img = env.reset()
    assert img.shape == (64, 64, 3)
    imgs = {k: NYCArena(bf, sf, res=64, seed=0, radius=200.0, episode_goal_dist=(30.0, 120.0), appearance=k).render(env.pos, env.heading) for k in ("clear", "fog", "night")}
    assert imgs["clear"].mean() > imgs["night"].mean()                          # night is darker
    assert not np.array_equal(imgs["clear"], imgs["fog"])
    s0 = env.state(); o, _ = env.step(1, render=False); assert o is None
    env.set_state(s0); assert np.allclose(env.pos, s0["pos"]) and env.collisions == s0["collisions"]
    # drive into a building: collisions count, position does not change
    env.pos = np.array([-70.0, -70.0 - 3.0]); env.heading = np.pi / 2          # 3 m south of the building at (-90..-50, -90..-50)? inside gap -> step north
    before = env.collisions; p = env.pos.copy()
    for _ in range(3): env.step(1, render=False)
    assert env.collisions > before and np.linalg.norm(env.pos - p) < 3.5
    assert env.shortest_path_length() > 0 and isinstance(env.obstacle_ahead(), bool)


def test_split_geo_blocks_are_disjoint(tmp_path):
    rng = np.random.default_rng(0); files = []
    for loc in range(2):
        E, T = 60, 24
        start = rng.uniform(-300, 300, (E, 2)) + loc * 5000
        pos = start[:, None] + np.cumsum(rng.normal(0, 3, (E, T, 2)), 1)
        d = dict(obs=np.zeros((E, T, 8, 32, 3), np.uint8), act=np.zeros((E, T), np.int8), pos=pos, heading=np.zeros((E, T)),
                 z=np.zeros((E, T, 16), np.float16), goal=start + 50, loc=np.full(E, loc))
        f = tmp_path / f"raw{loc}.npz"; np.savez_compressed(f, **d); files.append(str(f))
    out = tmp_path / "nyc"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "split_geo.py"), *files, "--out", str(out), "--block", "120", "--margin", "0"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr
    meta = json.load(open(str(out) + "_split.json")); tr, te = np.load(str(out) + "_train.npz"), np.load(str(out) + "_test.npz")
    assert meta["per_split"]["train"] + meta["per_split"]["val"] + meta["per_split"]["test"] == 120
    assert 0.55 < meta["per_split"]["train"] / 120 < 0.85 and meta["per_split"]["test"] > 0
    # whole blocks: no test start shares a block with a train start
    key = lambda p, l: (int(l), int(np.floor(p[0] / 120)), int(np.floor(p[1] / 120)))
    trk = {key(p, l) for p, l in zip(tr["pos"][:, 0], tr["loc"])}; tek = {key(p, l) for p, l in zip(te["pos"][:, 0], te["loc"])}
    assert not (trk & tek)
