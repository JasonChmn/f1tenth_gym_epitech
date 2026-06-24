"""
record_episode.py — Headless MP4 recording via matplotlib Agg + imageio-ffmpeg.

Renders every step with the top-down matplotlib view and writes a 20–30 fps
MP4 to ./recordings/episode_<timestamp>.mp4.  No system ffmpeg required —
imageio-ffmpeg bundles its own encoder.

Usage:
  docker compose run --rm viz python record_episode.py [--steps 5000]
  docker compose run --rm viz python record_episode.py --model submission/model.onnx
"""
import argparse
import os
import math
import time

import numpy as np

parser = argparse.ArgumentParser(description="Crash&Learn MP4 recorder")
parser.add_argument("--steps",    type=int,   default=5000)
parser.add_argument("--num-cars", type=int,   default=1)
parser.add_argument("--model",    type=str,   default="",
                    help="ONNX model for car 0 (random policy otherwise)")
parser.add_argument("--fps",      type=int,   default=20)
parser.add_argument("--output",   type=str,   default="",
                    help="Override output MP4 path")
args = parser.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import imageio

from env_simulation import (
    reset, get_obs, apply_action, simulation_step, get_step_info, close,
    _get_sim,
)

# ---------------------------------------------------------------------------
_REPO      = os.path.dirname(os.path.abspath(__file__))
_MAP_YAML  = os.path.join(_REPO, "examples", "example_map.yaml")
_MAP_IMG   = os.path.join(_REPO, "examples", "example_map.png")
_LIDAR_RAYS = 100
_LIDAR_FOV  = 4.7
_CAR_HALF_L = 0.29
_CAR_HALF_W = 0.155
_COLORS     = ["#FF6B35", "#4CC9F0", "#7BF178", "#F72585"]


def _load_map():
    with open(_MAP_YAML) as f:
        meta = yaml.safe_load(f)
    res = float(meta["resolution"])
    ox  = float(meta["origin"][0])
    oy  = float(meta["origin"][1])
    img = np.array(Image.open(_MAP_IMG))
    if img.ndim == 3:
        img = img[..., 0]
    return img, res, ox, oy


def _random_policy(obs, step_i):
    return 3.0, 0.08 * math.sin(step_i * 0.12)


def _onnx_policy(session, obs):
    feat = np.concatenate([
        obs["lidar"].astype(np.float32),
        np.array([obs["velocity"], obs["steering"], obs["progress"]], dtype=np.float32),
    ]).reshape(1, -1)
    inp = session.get_inputs()[0].name
    out = session.run(None, {inp: feat})[0][0]
    return float(out[0]), float(out[1])


def _draw_car(ax, x, y, theta, color):
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = np.array([
        [ _CAR_HALF_L,  _CAR_HALF_W],
        [ _CAR_HALF_L, -_CAR_HALF_W],
        [-_CAR_HALF_L, -_CAR_HALF_W],
        [-_CAR_HALF_L,  _CAR_HALF_W],
    ])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    world = corners @ R.T + np.array([x, y])
    poly = plt.Polygon(world, closed=True, fc=color, ec="white", lw=0.8, alpha=0.9, zorder=4)
    ax.add_patch(poly)
    dx, dy = 0.45 * cos_t, 0.45 * sin_t
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2), zorder=5)


def _draw_lidar(ax, x, y, theta, scan):
    n    = len(scan)
    incr = _LIDAR_FOV / max(n - 1, 1)
    for i, r in enumerate(scan):
        angle = theta - _LIDAR_FOV / 2.0 + i * incr
        t     = 1.0 - abs((i / max(n - 1, 1)) - 0.5) * 2.0
        alpha = 0.12 + 0.45 * t
        ax.plot([x, x + r * math.cos(angle)],
                [y, y + r * math.sin(angle)],
                color="#00FF88", lw=0.3, alpha=alpha, zorder=3)


def _render_frame(ax, fig, map_img_flipped, extent, num_cars, info, obs_list, poses,
                  step_i, total_steps):
    ax.clear()
    ax.set_facecolor("#0A1428")
    fig.patch.set_facecolor("#0A1428")

    ax.imshow(map_img_flipped, extent=extent, origin="lower",
              cmap="gray", alpha=0.55, zorder=1)

    for car_id in range(num_cars):
        if info["agent_status"][car_id] == 0:
            continue
        x, y, th = poses[car_id]
        color = _COLORS[car_id % len(_COLORS)]
        _draw_car(ax, x, y, th, color)
        if car_id == 0 and obs_list:
            _draw_lidar(ax, x, y, th, obs_list[car_id]["lidar"])

    obs0  = obs_list[0] if obs_list else {}
    lines = [
        f"step {info['step_count']:4d}/{total_steps}  t={info['time_elapsed']:6.1f}s",
        f"v={obs0.get('velocity', 0.0):+.2f}m/s  progress={obs0.get('progress', 0.0):.3f}",
        f"lap={obs0.get('lap_count', 0)}  friction={info['friction_current']:.3f}",
    ]
    if args.model:
        lines.append(f"model: {os.path.basename(args.model)}")

    ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=7, color="white", family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65), zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.axis("off")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3]  # RGB uint8


def main():
    map_img, res, ox, oy = _load_map()
    h, w = map_img.shape[:2]
    extent          = [ox, ox + w * res, oy, oy + h * res]
    map_img_flipped = np.flipud(map_img)

    session = None
    if args.model:
        import onnxruntime
        session = onnxruntime.InferenceSession(args.model)

    os.makedirs(os.path.join(_REPO, "recordings"), exist_ok=True)
    out_path = (args.output
                or os.path.join(_REPO, "recordings", f"episode_{int(time.time())}.mp4"))

    num_cars = max(1, min(args.num_cars, 4))
    reset(num_cars=num_cars)
    sim_obj = _get_sim()

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tight_layout(pad=0.4)

    # macro_block_size=None disables automatic resizing to codec block boundaries
    writer  = imageio.get_writer(out_path, fps=args.fps, codec="libx264", quality=8,
                                 macro_block_size=None)
    t_start = time.perf_counter()

    try:
        for step_i in range(args.steps):
            obs_list = [get_obs(i) for i in range(num_cars)]
            for car_id, obs in enumerate(obs_list):
                if session is not None and car_id == 0:
                    speed, steer = _onnx_policy(session, obs)
                else:
                    speed, steer = _random_policy(obs, step_i)
                apply_action(car_id, speed, steer)

            simulation_step()
            info = get_step_info()

            poses = []
            for car_id in range(num_cars):
                agent = sim_obj._sim.agents[car_id]
                poses.append((
                    float(agent.state[0]),
                    float(agent.state[1]),
                    float(agent.state[4]),
                ))

            frame = _render_frame(ax, fig, map_img_flipped, extent, num_cars,
                                  info, obs_list, poses, step_i, args.steps)
            writer.append_data(frame)

            if info["lap_complete"].get(0, False):
                print(f"  Lap complete at step {step_i + 1}, stopping early.")
                break

    finally:
        writer.close()

    plt.close(fig)
    close()

    elapsed = time.perf_counter() - t_start
    print(f"Saved: {out_path}  ({step_i + 1} frames in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
