"""
viz_pyglet.py — f1tenth native pyglet/OpenGL renderer (viz-gpu service only).

Requires nvidia-docker + DISPLAY.  CPU-only laptops must use viz_topdown.py.

Why the import order matters:
  env_simulation._bootstrap_gym_stubs() stubs pyglet into sys.modules so the
  engine loads without pyglet installed.  The stub check is guarded by
  `if name not in sys.modules`, so importing REAL pyglet here first prevents
  the stub from replacing it.  This file must therefore import pyglet before
  importing env_simulation (or anything that triggers env_simulation).

docker compose up viz-gpu   # requires: nvidia-docker, DISPLAY, /tmp/.X11-unix
"""

# Real pyglet must be imported BEFORE env_simulation stubs it.
import pyglet          # noqa: E402  (intentional import order)
import pyglet.gl       # noqa: E402

import os
import sys
import math
import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "gym"))

# env_simulation sees pyglet already in sys.modules → skips the stub
from env_simulation import (
    reset, get_obs, apply_action, simulation_step, get_step_info, close,
    _get_sim,
)
from f110_gym.envs.rendering import EnvRenderer

_MAP_YAML_BASE = os.path.join(_REPO, "examples", "example_map")
_MAP_EXT       = ".png"
_DECISION_HZ   = 20


def _random_policy(obs, counter):
    return 3.0, 0.08 * math.sin(counter[0] * 0.12)


def main():
    reset(num_cars=1)
    sim_obj = _get_sim()

    renderer = EnvRenderer(1000, 800)
    renderer.update_map(_MAP_YAML_BASE, _MAP_EXT)

    step_counter = [0]

    def update(dt):
        obs = get_obs(0)
        speed, steer = _random_policy(obs, step_counter)
        apply_action(0, speed, steer)
        simulation_step()

        # Push updated poses to renderer
        n      = sim_obj._num_agents
        poses  = np.zeros((n, 3))
        scans  = [None] * n
        for i in range(n):
            agent = sim_obj._sim.agents[i]
            poses[i] = [float(agent.state[0]),
                        float(agent.state[1]),
                        float(agent.state[4])]
            scans[i] = np.array(agent.scan, dtype=np.float32)

        renderer.poses    = poses
        renderer.vertices = None  # renderer recomputes from poses

        info = get_step_info()
        renderer.score_label.text = (
            f"step {info['step_count']}  "
            f"v={get_obs(0)['velocity']:+.2f}m/s  "
            f"lap={get_obs(0)['lap_count']}"
        )
        step_counter[0] += 1

    pyglet.clock.schedule_interval(update, 1.0 / _DECISION_HZ)
    pyglet.app.run()
    close()


if __name__ == "__main__":
    main()
