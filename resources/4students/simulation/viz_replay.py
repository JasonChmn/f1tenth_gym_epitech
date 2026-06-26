"""
viz_replay.py — Offline top-down replay from a .npz episode file.

No sim dependency. Requires X11/display.

Usage:
    python viz_replay.py [--npz episode.npz] [--map examples/example_map.yaml] [--follow]

Controls:
    Play/Pause button — real-time playback (uses sim_dt)
    Left/Right arrow  — step by 1
    Shift+Left/Right  — step by 10
    Slider            — seek to any step
"""
import argparse
import glob
import math
import os
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, Button
import numpy as np
from PIL import Image
import yaml

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
RECORD_DIR = "/app/recordings"
_maps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "examples")
_default_map = os.path.join(_maps_dir, "example_map.yaml") if os.path.exists(_maps_dir) else None

parser = argparse.ArgumentParser(description="Replay a recorded episode")
parser.add_argument("--npz", default=None, help="Episode NPZ (default: latest in /app/recordings/)")
parser.add_argument("--map", default=_default_map, help="Map YAML (auto-detected from episode if available)")
parser.add_argument("--follow", action="store_true", help="Re-center view on car 0 at each step")
args = parser.parse_args()

if args.npz is None:
    existing = sorted(glob.glob(os.path.join(RECORD_DIR, "episode_*.npz")))
    if not existing:
        sys.exit("No episode files in /app/recordings/. Use --npz <file>.")
    args.npz = existing[-1]
    print(f"Loading latest episode: {args.npz}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LIDAR_FOV  = 4.7
_CAR_HALF_L = 0.29
_CAR_HALF_W = 0.155
_COLORS     = ["#FF6B35", "#4CC9F0", "#7BF178", "#F72585"]
_BG         = "#0D0D0D"

# ---------------------------------------------------------------------------
# Load episode
# ---------------------------------------------------------------------------
print(f"Loading {args.npz} ...")
ep = np.load(args.npz)
lidar     = ep["lidar"]       # (T, C, 100)
velocity  = ep["velocity"]    # (T, C)
steering  = ep["steering"]    # (T, C)
progress  = ep["progress"]    # (T, C)
lap_count = ep["lap_count"]   # (T, C)
poses     = ep["poses"]       # (T, C, 3)  x y theta
actions   = ep["actions"]     # (T, C, 2)  spd steer
status    = ep["status"]      # (T, C)
friction  = ep["friction"]    # (T,)
max_prog  = ep["max_prog"]    # (T, C)
sim_dt    = float(ep["sim_dt"])
num_cars  = int(ep["num_cars"])
T         = lidar.shape[0]
print(f"  {T} steps, {num_cars} car(s), sim_dt={sim_dt}s")

# ---------------------------------------------------------------------------
# Resolve map (NPZ map_name overrides --map if file exists)
# ---------------------------------------------------------------------------
map_yaml = args.map
if "map_name" in ep:
    auto = os.path.join(_maps_dir, f"{str(ep['map_name']).strip()}.yaml")
    if os.path.exists(auto):
        map_yaml = auto
        print(f"Auto-detected map: {auto}")

if map_yaml is None:
    sys.exit("No map found. Use --map <path>.")

with open(map_yaml) as f:
    meta = yaml.safe_load(f)
res = float(meta["resolution"])
ox, oy = float(meta["origin"][0]), float(meta["origin"][1])

img = np.array(Image.open(map_yaml.replace(".yaml", ".png")))
if img.ndim == 3:
    img = img[..., 0]
h, w = img.shape[:2]
extent = [ox, ox + w * res, oy, oy + h * res]

half_win = max(extent[1] - extent[0], extent[3] - extent[2]) / 4.0  # follow half-window

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _car_vertices(x, y, theta):
    c, s = math.cos(theta), math.sin(theta)
    corners = np.array([
        [ _CAR_HALF_L,  _CAR_HALF_W],
        [ _CAR_HALF_L, -_CAR_HALF_W],
        [-_CAR_HALF_L, -_CAR_HALF_W],
        [-_CAR_HALF_L,  _CAR_HALF_W],
    ])
    R = np.array([[c, -s], [s, c]])
    return corners @ R.T + (x, y)


def _lidar_segments(x, y, theta, scan):
    """(N, 2, 2) for LineCollection."""
    n = len(scan)
    angles = theta - _LIDAR_FOV / 2.0 + np.arange(n) * (_LIDAR_FOV / max(n - 1, 1))
    seg = np.empty((n, 2, 2))
    seg[:, 0, 0] = x
    seg[:, 0, 1] = y
    seg[:, 1, 0] = x + scan * np.cos(angles)
    seg[:, 1, 1] = y + scan * np.sin(angles)
    return seg

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor(_BG)

ax_map = fig.add_axes([0.01, 0.08, 0.98, 0.90])   # map: nearly full window
ax_tel = fig.add_axes([0.01, 0.69, 0.22, 0.30])   # telemetry overlay (bottom-left)
ax_sld = fig.add_axes([0.01, 0.03, 0.79, 0.035])  # slider
ax_btn = fig.add_axes([0.83, 0.025, 0.14, 0.045]) # play/pause button

for ax in (ax_map, ax_tel):
    ax.set_facecolor(_BG)
    ax.axis("off")

ax_map.set_xlim(extent[0], extent[1])
ax_map.set_ylim(extent[2], extent[3])
ax_map.set_aspect("equal")
ax_map.imshow(np.flipud(img), extent=extent, origin="lower",
              cmap="gray_r", alpha=0.85, zorder=1, interpolation="nearest")

if False:
    # static full trajectories (last point excluded, intentional)
    for cid in range(num_cars):
        ax_map.plot(poses[:-1, cid, 0], poses[:-1, cid, 1],
                    color=_COLORS[cid % len(_COLORS)], lw=2.0, alpha=0.5, zorder=2)
else:
    for cid in range(num_cars):
        color = _COLORS[cid % len(_COLORS)]
        # Only plot up to the car's last valid step (where status != 0)
        valid_steps = np.where(status[:, cid] != 0)[0]
        if len(valid_steps) > 0:
            last_valid = valid_steps[-1]
            if last_valid > 0:
                ax_map.plot(poses[:last_valid, cid, 0], poses[:last_valid, cid, 1],
                            color=color, lw=2.0, alpha=0.5, zorder=2)

# dynamic artists
car_polys, car_arrows, traj_markers = [], [], []
for cid in range(num_cars):
    color = _COLORS[cid % len(_COLORS)]
    poly = Polygon(np.zeros((4, 2)), closed=True, fc=color, ec="white",
                   lw=0.8, alpha=0.9, zorder=4)
    ax_map.add_patch(poly)
    arrow, = ax_map.plot([], [], color="white", lw=1.2, zorder=5, solid_capstyle="butt")
    mk, = ax_map.plot([], [], "o", color=color, ms=7, zorder=6,
                      markeredgecolor="white", markeredgewidth=0.8)
    car_polys.append(poly)
    car_arrows.append(arrow)
    traj_markers.append(mk)

lidar_lc = LineCollection([], color="#00BB55", lw=1.5, alpha=0.7, zorder=3)
ax_map.add_collection(lidar_lc)

tel_text = ax_tel.text(0.05, 0.98, "", transform=ax_tel.transAxes, va="top", ha="left",
                       fontsize=13, color="white", family="monospace",
                       bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))

step_label = ax_map.text(0.01, 0.99, "", transform=ax_map.transAxes, va="top", ha="left",
                         fontsize=10, color="white", family="monospace", zorder=7,
                         bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65))

step_num_label = ax_map.text(0.01, 0.94, "0", transform=ax_map.transAxes, va="top", ha="left",
                             fontsize=12, fontweight="bold", color="#4CC9F0", family="monospace", zorder=7,
                             bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.65))

slider = Slider(ax_sld, "Step", 0, T - 1, valinit=0, valstep=1,
                color="#1A6FD4", track_color="#CCCCCC")
ax_sld.xaxis.label.set_color("white")
ax_sld.tick_params(colors="white")

button = Button(ax_btn, "\u25b6  Play", color="#444444", hovercolor="#555555")
button.label.set_color("white")
button.label.set_fontweight("bold")
_playing = False

# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------
def _update(step_i):
    step_i = int(np.clip(step_i, 0, T - 1))

    for cid in range(num_cars):
        x, y, theta = poses[step_i, cid]
        if status[step_i, cid] != 0:
            c, s = math.cos(theta), math.sin(theta)
            car_polys[cid].set_xy(_car_vertices(x, y, theta))
            car_polys[cid].set_visible(True)
            car_arrows[cid].set_data([x, x + 0.45 * c], [y, y + 0.45 * s])
            car_arrows[cid].set_visible(True)
            traj_markers[cid].set_data([x], [y])
        else:
            car_polys[cid].set_visible(False)
            car_arrows[cid].set_visible(False)
            traj_markers[cid].set_data([], [])

    x0, y0, th0 = poses[step_i, 0]
    lidar_lc.set_segments(_lidar_segments(x0, y0, th0, lidar[step_i, 0]))

    if args.follow:
        ax_map.set_xlim(x0 - half_win, x0 + half_win)
        ax_map.set_ylim(y0 - half_win, y0 + half_win)

    step_label.set_text(f"t={step_i * sim_dt:.2f}s")
    step_num_label.set_text(f"Step {step_i}")

    lines = []
    for cid in range(num_cars):
        st = "active" if status[step_i, cid] != 0 else "DNF"
        lines += [
            f"─── car {cid} ({st}) ───", "",
            f"velocity : {velocity[step_i, cid]:+.2f} m/s",
            f"steering : {steering[step_i, cid]:+.4f} rad",
            f"cmd speed: {actions[step_i, cid, 0]:+.2f}",
            f"cmd steer: {actions[step_i, cid, 1]:+.4f}", "",
            f"progress : {progress[step_i, cid]:.3f}",
            f"max prog : {max_prog[step_i, cid]:.3f}",
            f"lap      : {lap_count[step_i, cid]}", "",
        ]
    lines.append(f"friction : {friction[step_i]:.3f}")
    tel_text.set_text("\n".join(lines))

    fig.canvas.draw_idle()


slider.on_changed(_update)

# ---------------------------------------------------------------------------
# Play / pause toggle
# ---------------------------------------------------------------------------
def _toggle(event=None):
    global _playing
    _playing = not _playing
    if _playing:
        if int(slider.val) >= T - 1:
            slider.set_val(0)
        button.label.set_text("\u23f8  Pause")
        button.ax.set_facecolor("#1A6FD4")
    else:
        button.label.set_text("\u25b6  Play")
        button.ax.set_facecolor("#444444")
    fig.canvas.draw_idle()

button.on_clicked(_toggle)

# ---------------------------------------------------------------------------
# Keyboard — pauses on manual step
# ---------------------------------------------------------------------------
def _on_key(event):
    global _playing
    step = {"left": -1, "right": 1, "shift+left": -10, "shift+right": 10}.get(event.key)
    if step is None:
        return
    if _playing:
        _toggle()
    slider.set_val(int(np.clip(slider.val + step, 0, T - 1)))

fig.canvas.mpl_connect("key_press_event", _on_key)

# ---------------------------------------------------------------------------
# Main loop — flush_events is X11-friendly, no timer/thread
# ---------------------------------------------------------------------------
_update(0)
plt.show(block=False)

while plt.fignum_exists(fig.number):
    if _playing:
        v = int(slider.val)
        if v >= T - 1:
            _toggle()   # auto-stop at end
        else:
            slider.set_val(v + 1)   # triggers _update via on_changed
    fig.canvas.flush_events()