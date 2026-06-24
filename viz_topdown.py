"""
viz_topdown.py — Visualisation top-down Matplotlib de la simulation.

Affichage interactif (TkAgg, requiert DISPLAY) — calé sur le temps réel (RT 1x) :
   python viz_topdown.py [--steps N] [--label TEXT] [--view-mode follow]

Headless (ni affichage ni sauvegarde) :
   python viz_topdown.py --no-display [--steps N] [--num-cars N] [--model X.onnx]

Enregistrement MP4 (vidéo lue en temps réel à --render-fps) :
   python viz_topdown.py --record episode.mp4 [--steps N]

Via Docker :
   xhost +local:docker
   docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\
     viz python viz_topdown.py --label "random agent"
   docker compose run --rm -v .:/app viz python viz_topdown.py --record /app/episode.mp4 --steps 500
   docker compose run --rm viz python viz_topdown.py --no-display --steps 500
"""
import argparse
import math
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="Crash&Learn top-down visualizer")
_parser.add_argument("--no-display", action="store_true",
                     default=not bool(os.environ.get("DISPLAY", "")),
                     help="Mode headless — ni affichage ni sauvegarde")
_parser.add_argument("--steps", type=int, default=500)
_parser.add_argument("--num-cars", type=int, default=1)
_parser.add_argument("--label", type=str, default="")
_parser.add_argument("--model", type=str, default="",
                     help="Modèle ONNX pour la voiture 0 (politique aléatoire sinon)")
_parser.add_argument("--record", type=str, default="",
                     help="Chemin du fichier MP4 à enregistrer")
_parser.add_argument("--view-mode", type=str, default="follow",
                     choices=["full", "follow", "interactive"],
                     help="'full'=circuit entier, 'follow'=centré sur la voiture, "
                          "'interactive'=zoom/pan libre avec auto-centrage")
_parser.add_argument("--zoom-scale", type=float, default=2.0,
                     help="Facteur de zoom (follow/interactive) : 2.0 ⇒ ~moitié du circuit visible")
_parser.add_argument("--preview-steps", type=int, default=30,
                     help="N steps en vue circuit entier avant de passer en mode follow (défaut 30)")
_parser.add_argument("--render-fps", type=float, default=30.0,
                      help="Cadence de rendu / vidéo (défaut 30)")
_parser.add_argument("--render-every", type=int, default=0,
                      help="Render 1 step out of N (0 = derive from --render-fps and _SIM_DT)")
_parser.add_argument("--fontsize", type=int, default=11,
                     help="Taille de police de l'overlay (défaut 11)")
_parser.add_argument("--print-time", action="store_true",
                     help="Print per-stage timing to analyze RT ratio issues")
args = _parser.parse_args()

import matplotlib
matplotlib.use("Agg" if args.no_display else "TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from PIL import Image
import yaml

from env_simulation import (
    reset, get_obs, apply_action, simulation_step, get_step_info, close,
    _get_sim,
)

# ---------------------------------------------------------------------------
# Constantes (miroir de env_simulation — visibilité côté instructeur assumée)
# ---------------------------------------------------------------------------
_REPO       = os.path.dirname(os.path.abspath(__file__))
_MAP_YAML   = os.path.join(_REPO, "examples", "example_map.yaml")
_MAP_IMG    = os.path.join(_REPO, "examples", "example_map.png")
_LIDAR_FOV  = 4.7    # rad
_CAR_HALF_L = 0.29   # m
_CAR_HALF_W = 0.155  # m
_COLORS     = ["#FF6B35", "#4CC9F0", "#7BF178", "#F72585"]  # ego, opp1..3
_SIM_DT     = 0.02   # pas physique (s) — doit matcher env_simulation
_BG         = "#0A1428"


def _load_map():
    with open(_MAP_YAML) as f:
        meta = yaml.safe_load(f)
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    img = np.array(Image.open(_MAP_IMG))
    if img.ndim == 3:
        img = img[..., 0]
    return img, res, ox, oy


# ---------------------------------------------------------------------------
# Politiques
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Géométrie (pures)
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
    return corners @ R.T + np.array([x, y])


def _lidar_segments(x, y, theta, scan):
    """(N, 2, 2) pour LineCollection."""
    n = len(scan)
    incr = _LIDAR_FOV / max(n - 1, 1)
    angles = theta - _LIDAR_FOV / 2.0 + np.arange(n) * incr
    ex = x + scan * np.cos(angles)
    ey = y + scan * np.sin(angles)
    seg = np.empty((n, 2, 2))
    seg[:, 0, 0] = x
    seg[:, 0, 1] = y
    seg[:, 1, 0] = ex
    seg[:, 1, 1] = ey
    return seg


# ---------------------------------------------------------------------------
# Renderer — artistes persistants, mis à jour en place (pas de ax.clear)
# ---------------------------------------------------------------------------
class Renderer:
    def __init__(self, fig, ax, map_img_flipped, extent, num_cars):
        self.fig, self.ax = fig, ax
        self.extent = extent
        self.num_cars = num_cars

        # Demi-largeur de fenêtre constante en mode follow/interactive.
        map_w = max(extent[1] - extent[0], extent[3] - extent[2])
        self.half_zoom = map_w / (2.0 * args.zoom_scale)  # <-- /2 : sinon plein cadre

        # Caméra (mode interactive).
        self.center = None
        self.manual_override = False

        ax.set_facecolor(_BG)
        fig.patch.set_facecolor(_BG)
        ax.set_aspect("equal")          # carrés carrés
        ax.set_position([0, 0, 1, 1])   # l'axe remplit la figure
        ax.axis("off")

        ax.imshow(map_img_flipped, extent=extent, origin="lower",
                  cmap="gray", alpha=0.55, zorder=1, interpolation="nearest")

        self.cars = []
        for cid in range(num_cars):
            color = _COLORS[cid % len(_COLORS)]
            poly = Polygon(np.zeros((4, 2)), closed=True, fc=color, ec="white",
                           lw=0.8, alpha=0.9, zorder=4, visible=False)
            ax.add_patch(poly)
            arrow = ax.plot([], [], color="white", lw=1.2, zorder=5,
                            solid_capstyle="butt")[0]
            self.cars.append((poly, arrow))

        # Un LineCollection lidar pour la voiture 0 seulement.
        self.lidar = LineCollection([], color="#00FF88", lw=0.3, alpha=0.35,
                                    zorder=3, visible=False)
        ax.add_collection(self.lidar)

        self.text = ax.text(
            0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
            fontsize=args.fontsize, color="white", family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65), zorder=6,
        )

    # --- caméra ------------------------------------------------------------
    def _apply_limits(self, poses, mode):
        if mode == "full":
            self.ax.set_xlim(self.extent[0], self.extent[1])
            self.ax.set_ylim(self.extent[2], self.extent[3])
            return

        cx, cy, _ = poses[0]
        if mode == "interactive" and self.manual_override:
            return  # l'utilisateur pilote la vue

        if mode == "interactive" and self.center is not None:
            cx, cy = self.center
        else:
            self.center = (cx, cy)

        h = self.half_zoom
        self.ax.set_xlim(cx - h, cx + h)
        self.ax.set_ylim(cy - h, cy + h)

    # --- frame -------------------------------------------------------------
    def update(self, info, obs_list, poses, sim_time, wall_elapsed, mode):
        has_obs = bool(obs_list)
        for cid in range(self.num_cars):
            poly, arrow = self.cars[cid]
            active = info["agent_status"][cid] != 0
            if active:
                x, y, th = poses[cid]
                poly.set_xy(_car_vertices(x, y, th))
                poly.set_visible(True)
                c, s = math.cos(th), math.sin(th)
                arrow.set_data([x, x + 0.45 * c], [y, y + 0.45 * s])
                arrow.set_visible(True)
                if has_obs and cid == 0:
                    self.lidar.set_segments(
                        _lidar_segments(x, y, th, obs_list[0]["lidar"]))
                    self.lidar.set_visible(True)
            else:
                poly.set_visible(False)
                arrow.set_visible(False)
                if cid == 0:
                    self.lidar.set_visible(False)

        rt = sim_time / wall_elapsed if wall_elapsed > 0 else 0.0
        obs0 = obs_list[0] if obs_list else {}
        lines = [
            f"step {info['step_count']:4d}  t={info['time_elapsed']:6.1f}s  RT×{rt:.1f}",
            f"v={obs0.get('velocity', 0.0):+.2f}m/s  steer={obs0.get('steering', 0.0):+.4f}rad",
            f"progress={obs0.get('progress', 0.0):.3f}  lap={obs0.get('lap_count', 0)}",
            f"friction={info['friction_current']:.3f}",
        ]
        if args.label:
            lines.append(args.label)
        if args.model:
            lines.append(f"model: {os.path.basename(args.model)}")
        self.text.set_text("\n".join(lines))

        self._apply_limits(poses, mode)

    def grab_frame(self):
        """Frame RGB (H, W, 3) uint8 — compatible Matplotlib >= 3.10."""
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[..., :3].copy()


# ---------------------------------------------------------------------------
# Enregistrement vidéo
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self, path, fps):
        import cv2  # local : optionnel
        self._cv2 = cv2
        self.path = path
        self.fps = fps
        self.writer = None
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def write(self, frame_rgb):
        cv2 = self._cv2
        if self.writer is None:
            h, w = frame_rgb.shape[:2]
            w -= w % 2  # dimensions paires requises par mp4v
            h -= h % 2
            self._wh = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
            print(f"Recording started at {self.fps} FPS -> {self.path}")
        w, h = self._wh
        frame = frame_rgb[:h, :w]
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f"Recorded: {self.path}")


# ---------------------------------------------------------------------------
# Timing helper (lightweight, no threading)
# ---------------------------------------------------------------------------
class _Timer:
    """Minimal wall-clock timer used when --print-time is enabled."""
    __slots__ = ("label", "t0")
    def __init__(self, label: str):
        self.label = label
        self.t0 = time.perf_counter()
    def elapsed(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0  # ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    map_img, res, ox, oy = _load_map()
    h, w = map_img.shape[:2]
    extent = [ox, ox + w * res, oy, oy + h * res]
    map_img_flipped = np.flipud(map_img)

    session = None
    if args.model:
        import onnxruntime
        session = onnxruntime.InferenceSession(args.model)

    num_cars = max(1, min(args.num_cars, 4))
    reset(num_cars=num_cars)
    sim_obj = _get_sim()

    recorder = None
    if args.record:
        try:
            recorder = Recorder(args.record, args.render_fps)
        except ImportError:
            print("Warning: opencv-python absent. --record ignoré.")
            args.record = ""

    live = not args.no_display and not args.record

    # Figure carrée : les limites follow/interactive sont carrées ⇒ l'axe
    # remplit sans bandes vides (le ratio 10x8 d'origine causait le scaling bugué).
    fig, ax = plt.subplots(figsize=(9, 9))
    renderer = Renderer(fig, ax, map_img_flipped, extent, num_cars)

    if live:
        plt.ion()
        plt.show(block=False)

        def _on_press(event):
            if event.inaxes == ax and args.view_mode == "interactive":
                renderer.manual_override = True

        fig.canvas.mpl_connect("button_press_event", _on_press)

    # Decimation factor: derive from --render-fps if --render-every not set
    _render_every = args.render_every if args.render_every > 0 \
        else max(1, round(1.0 / (args.render_fps * _SIM_DT)))

    sim_time = 0.0
    t_start = time.perf_counter()

    t_wall_prev = t_start
    have_printed_timing = False  # guard first-iteration dt calc

    for step_i in range(args.steps):
        wall_t_step_start = time.perf_counter()

        # --- observation ---------------------------------------------------
        _t_obs = _Timer("obs") if args.print_time else None
        obs_list = [get_obs(i) for i in range(num_cars)]
        if _t_obs is not None: have_printed_timing = True; _obs_el = _t_obs.elapsed()

        # --- policy --------------------------------------------------------
        _t_pol = _Timer("policy") if args.print_time else None
        for cid, obs in enumerate(obs_list):
            if session is not None and cid == 0:
                speed, steer = _onnx_policy(session, obs)
            else:
                speed, steer = _random_policy(obs, step_i)
            apply_action(cid, speed, steer)
        if _t_pol is not None: have_printed_timing = True; _pol_el = _t_pol.elapsed()

        # --- simulation ----------------------------------------------------
        _t_sim = _Timer("sim") if args.print_time else None
        simulation_step()
        sim_time += _SIM_DT
        info = get_step_info()
        if _t_sim is not None: have_printed_timing = True; _sim_el = _t_sim.elapsed()

        # --- pose extraction -----------------------------------------------
        poses = []
        for cid in range(num_cars):
            agent = sim_obj._sim.agents[cid]
            poses.append((float(agent.state[0]),
                          float(agent.state[1]),
                          float(agent.state[4])))

        # --- compute wall clock before render ------------------------------
        t_wall_now = time.perf_counter()
        wall_elapsed = t_wall_now - t_start

        # --- rendering -----------------------------------------------------
        _rendered_this_step = False
        if step_i % _render_every == 0:
            _t_upd = _Timer("render") if args.print_time else None
            mode = "full" if step_i < args.preview_steps else args.view_mode
            renderer.update(info, obs_list, poses, sim_time, wall_elapsed, mode)
            if _t_upd is not None: have_printed_timing = True; _upd_el = _t_upd.elapsed()

            # frame grab (record only) ------------------------------------
            if recorder is not None:
                _t_grab = _Timer("grab") if args.print_time else None
                recorder.write(renderer.grab_frame())
                if _t_grab is not None: have_printed_timing = True; _grab_el = _t_grab.elapsed()
            elif live:
                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(0.001)

            _rendered_this_step = True
        

        # --- timing summary (every loop iteration) -------------------------
        wall_t_step_end = time.perf_counter()
        wall_dt = (wall_t_step_end - t_wall_prev) * 1000.0  # ms per loop iteration
        t_wall_prev = wall_t_step_end

        if args.print_time:
            total_ms = (wall_t_step_end - wall_t_step_start) * 1000.0

            # build timing string
            parts = []
            if have_printed_timing:
                parts.append(f"obs={_obs_el:.1f}ms")
                parts.append(f"pol={_pol_el:.1f}ms")
                parts.append(f"sim={_sim_el:.1f}ms")
            if _rendered_this_step:
                parts.append(f"upd={_upd_el:.1f}ms")
                if recorder is not None:
                    parts.append(f"grab={_grab_el:.1f}ms")
            parts.append(f"wall_dt={wall_dt:.1f}ms")
            parts.append(f"total={total_ms:.1f}ms")
            # no sleep in this mode

            rt = sim_time / wall_elapsed if wall_elapsed > 0 else 0.0
            print(f"[timing] step={step_i:4d} {', '.join(parts)}  RT×{rt:.1f}", flush=True)

if __name__ == "__main__":
    main()