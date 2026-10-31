import numpy as np
from wm2cx.env import FlyArena, scripted_policy


def test_state_roundtrip_and_renderless_step():
    env = FlyArena(task="arrange", n_boxes=1, res=64, seed=3); env.reset(); s0 = env.state()
    img, ok = env.step(1, render=False); assert img is None
    env.set_state(s0)
    assert np.allclose(env.pos, s0["pos"]) and env.heading == s0["heading"] and np.allclose(env.box, s0["box"]) and env.t == s0["t"]
    img2, _ = env.step(1); assert img2.shape == (64, 64, 3)
    assert np.allclose(env.pos, s0["pos"] + 0.6 * np.array([np.cos(s0["heading"]), np.sin(s0["heading"])]))


def test_turn_noise_makes_turns_continuous():
    a = FlyArena(res=32, seed=0); b = FlyArena(res=32, seed=0, turn_noise=0.3)
    h0 = a.heading; a.step(2, render=False); b.step(2, render=False)
    assert np.isclose((a.heading - h0 + np.pi) % (2 * np.pi) - np.pi, np.pi / 8)
    assert not np.isclose((b.heading - h0 + np.pi) % (2 * np.pi) - np.pi, np.pi / 8)


def test_scripted_policy_reaches_goal():
    env = FlyArena(task="nav", res=32, seed=1); env.reset(); rng = np.random.default_rng(0)
    for _ in range(60):
        _, s = env.step(scripted_policy(env, rng, 0.0), render=False)
        if s: break
    assert s == 1.0
