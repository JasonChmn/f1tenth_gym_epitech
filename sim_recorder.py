"""
sim_recorder.py — Run a full episode at maximum speed, dump all obs/state to .npz.

Usage:
    python sim_recorder.py [--steps 500] [--num-cars 1] [--model agent.onnx] [--out episode.npz]
"""
import argparse
import math
import os
import glob
import time

import numpy as np

_SIM_DT = 0.02  # must match env_simulation

OUTPUT_DIR = "/app/recordings"

parser = argparse.ArgumentParser(description="Record a sim episode to .npz")
parser.add_argument("--steps",    type=int,   default=500)
parser.add_argument("--num-cars", type=int,   default=1)
parser.add_argument("--model",    type=str,   default="",  help="ONNX model for car 0")
parser.add_argument("--controller", type=str, default="random", choices=["random", "onnx", "pure_pursuit"],
                    help="Controller policy for car 0 (default: random)")
parser.add_argument("--out",      type=str,   default=None, help="Output path (default: recordings/episode_XXX.npz)")
args = parser.parse_args()

# Auto-generate episode ID if not provided
if args.out is None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(OUTPUT_DIR, "episode_*.npz"))
    episode_id = len(existing) + 1
    args.out = os.path.join(OUTPUT_DIR, f"episode_{episode_id:04d}.npz")

num_cars = max(1, min(args.num_cars, 4))
T = args.steps

# ---------------------------------------------------------------------------
# Policy / controller selection
# ---------------------------------------------------------------------------
_SPEED   = 2.0
_LOOKAHEAD = 1.0

from env_simulation import _PARAMS as _EP
_WHEELBASE = _EP["lf"] + _EP["lr"]
_SMIN, _SMAX = _EP["s_min"], _EP["s_max"]


def _random_policy(step_i):
    return _SPEED, 0.08 * math.sin(step_i * 0.12)


def _pure_pursuit_policy(step_i, obs, waypoints, arc, total_arc, state):
    """Pure pursuit: find lookahead waypoint by walking from closest point, compute steering as angle to carrot."""
    x, y, yaw = float(state[0]), float(state[1]), float(state[4])

    # Find closest waypoint on centerline
    dists = (waypoints[:, 0] - x) ** 2 + (waypoints[:, 1] - y) ** 2
    idx = int(np.argmin(dists))

    # Walk from closest point, accumulating distance until >= lookahead
    N = len(waypoints)
    cum = 0.0
    target_idx = idx  # fallback: closest point itself
    for i in range(1, N):
        wi = (idx + i) % N
        j = (idx + i - 1) % N
        dx_w = waypoints[wi, 0] - waypoints[j, 0]
        dy_w = waypoints[wi, 1] - waypoints[j, 1]
        cum += math.sqrt(dx_w * dx_w + dy_w * dy_w)
        if cum >= _LOOKAHEAD:
            target_idx = wi
            break

    tx, ty = waypoints[target_idx, 0], waypoints[target_idx, 1]

    # Angle from car's orientation to the carrot
    dx, dy = tx - x, ty - y
    angle_to_target = math.atan2(dy, dx)
    angle_diff = angle_to_target - yaw

    # Normalize to [-pi, pi]
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    # Use the angle directly as steering (clipped to bounds)
    steer = max(_SMIN, min(_SMAX, angle_diff))
    return _SPEED, steer


def _onnx_policy(session, obs):
    feat = np.concatenate([
        obs["lidar"].astype(np.float32),
        np.array([obs["velocity"], obs["steering"], obs["progress"]], dtype=np.float32),
    ]).reshape(1, -1)
    inp = session.get_inputs()[0].name
    out = session.run(None, {inp: feat})[0][0]
    return float(out[0]), float(out[1])


session = None
if args.model:
    import onnxruntime
    session = onnxruntime.InferenceSession(args.model)
    print(f"Loaded ONNX model: {args.model}")

# ---------------------------------------------------------------------------
# Sim import
# ---------------------------------------------------------------------------
from env_simulation import (
    reset, get_obs, apply_action, simulation_step, get_step_info, close, _get_sim,
    get_current_map,
)

# ---------------------------------------------------------------------------
# Pre-allocate
# ---------------------------------------------------------------------------
lidar     = np.zeros((T, num_cars, 100), dtype=np.float32)
velocity  = np.zeros((T, num_cars),      dtype=np.float32)
steering  = np.zeros((T, num_cars),      dtype=np.float32)
progress  = np.zeros((T, num_cars),      dtype=np.float32)
lap_count = np.zeros((T, num_cars),      dtype=np.int32)
poses     = np.zeros((T, num_cars, 3),   dtype=np.float32)  # x, y, theta
actions   = np.zeros((T, num_cars, 2),   dtype=np.float32)  # speed, steer
status    = np.zeros((T, num_cars),      dtype=np.int32)
friction  = np.zeros((T,),               dtype=np.float32)
max_prog  = np.zeros((T, num_cars),      dtype=np.float32)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
reset(num_cars=num_cars)
sim_obj = _get_sim()

t_start = time.perf_counter()
actual_steps = 0
prev_obs = None  # observations from previous step (used for action inputs)

for step_i in range(T):
    # --- controller helpers (pure_pursuit needs waypoints/state from sim_obj) ---
    use_pp = args.controller == "pure_pursuit"
    wpts = sim_obj._waypoints if use_pp else None
    arc  = sim_obj._arc      if use_pp else None
    tac  = sim_obj._total_arc if use_pp else None

    # --- controller: pick speed/steer from prev_obs (or skip on step 0) ---
    decisions = {}  # {cid: (spd, steer)}
    if prev_obs is not None:
        for cid, obs in enumerate(prev_obs):
            spd, steer = _random_policy(step_i)
            if args.controller == "onnx" and cid == 0 and session is not None:
                spd, steer = _onnx_policy(session, obs)
            elif args.controller == "pure_pursuit" and cid == 0 and use_pp:
                state = sim_obj._sim.agents[cid].state
                spd, steer = _pure_pursuit_policy(step_i, obs, wpts, arc, tac, state)
            decisions[cid] = (spd, steer)

    # --- actions (apply controller decisions; skip on step 0 — car starts idle) ---
    for cid, (spd, steer) in decisions.items():
        apply_action(cid, spd, steer)
        actions[step_i, cid, 0] = spd
        actions[step_i, cid, 1] = steer

    # --- step (THIS is where lap_count increments and DNF can fire) ---
    simulation_step()
    info = get_step_info()

    # --- obs — capture AFTER step so lap_count reflects any completed laps ---
    obs_list = [get_obs(i) for i in range(num_cars)]

    # --- store obs (all from post-step observations, consistent source) ---
    for cid, obs in enumerate(obs_list):
        lidar[step_i, cid]     = obs["lidar"]
        velocity[step_i, cid]  = obs["velocity"]
        steering[step_i, cid]  = obs["steering"]
        progress[step_i, cid]  = obs["progress"]
        lap_count[step_i, cid] = obs["lap_count"]
        status[step_i, cid]    = info["agent_status"][cid]

    # --- poses (from sim state after step) ---
    for cid in range(num_cars):
        agent = sim_obj._sim.agents[cid]
        poses[step_i, cid, 0] = float(agent.state[0])
        poses[step_i, cid, 1] = float(agent.state[1])
        poses[step_i, cid, 2] = float(agent.state[4])

    friction[step_i] = info["friction_current"]
    for cid in range(num_cars):
        max_prog[step_i, cid] = float(info["max_progress"][cid])

    actual_steps = step_i + 1

    # --- save obs for next iteration's controller inputs ---
    prev_obs = obs_list

    # --- progress print ---
    if actual_steps % 100 == 0:
        elapsed = time.perf_counter() - t_start
        rt = (actual_steps * _SIM_DT) / elapsed
        print(f"step={actual_steps:4d}  sim_t={actual_steps*_SIM_DT:.1f}s  "
              f"wall={elapsed:.2f}s  RT×{rt:.1f}")

    # --- early exit if all DNF ---
    if all(info["agent_status"][i] == 0 for i in range(num_cars)):
        print(f"All cars DNF at step {actual_steps}")
        break

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
wall_total = time.perf_counter() - t_start
rt_total = (actual_steps * _SIM_DT) / wall_total

np.savez_compressed(
    args.out,
    lidar     = lidar[:actual_steps],
    velocity  = velocity[:actual_steps],
    steering  = steering[:actual_steps],
    progress  = progress[:actual_steps],
    lap_count = lap_count[:actual_steps],
    poses     = poses[:actual_steps],
    actions   = actions[:actual_steps],
    status    = status[:actual_steps],
    friction  = friction[:actual_steps],
    max_prog  = max_prog[:actual_steps],
    sim_dt    = np.float32(_SIM_DT),
    num_cars  = np.int32(num_cars),
    map_name  = np.str_(get_current_map()),
)

close()
print(f"\nDone. {actual_steps} steps in {wall_total:.2f}s  RT×{rt_total:.1f}")
print(f"Saved → {args.out}")
