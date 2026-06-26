# Crash&Learn Grand Prix — Wrapper Technical Design

> Authoritative reference for the fork implementation.
> Claude Code must read this file before modifying any code.
> Last updated: 2026-06-24. Update date on every major revision.
> Cross-referenced with: 1_env_simulation_specs.md, 1_review_after_refactor.md, class_diagram.md, recon.md

---

## Overview

Opaque wrapper on top of the **f1tenth_gym engine** exposing a minimal API to students.
Wrapper source code is not distributed — only `env_simulation.py` is visible.

**Backend migration (2026-06):** the simulation backend moved from `racecar_gym` / PyBullet
(3D rigid-body) to the **f1tenth_gym engine** (2D, single-track dynamic model + numba laser
scan). The public API and the student-facing contract are **unchanged** — only the internals
of `_Sim` were swapped.

We drive the engine **directly** through `f110_gym.envs.base_classes.Simulator` and the
laser/dynamic/collision models. We **bypass f1tenth's gym shell** (`F110Env`, `gym.make`)
entirely. Consequences of bypassing the shell, all intentional:

- **No `gym` / `gymnasium` dependency is installed for the simulator.** The engine modules
  (`base_classes`, `laser_models`, `dynamic_models`, `collision_models`) are pure
  numpy/numba/scipy. f1tenth's `setup.py` is NOT used (it pins `gym==0.19.0` and
  `numpy<=1.22`, which break on modern Python). Engine modules are imported by path; only
  real runtime deps are installed (numpy, numba, scipy, Pillow, pyyaml).
- **No `done`-on-collision.** That logic lives in `F110Env._check_done`, which we do not use.
  Termination is fully the student's responsibility (see below). This is what makes
  "collisions are not punished" essentially free — see the Collision policy section.
- **No pyglet renderer on the default import path.** f1tenth's native renderer is pyglet/GL
  and is attached to `F110Env`. Our default visualization is matplotlib (headless-capable);
  the pyglet renderer remains available as an **opt-in GPU extra** (see Visualization).

Internally, the wrapper is a private singleton `_Sim` wrapping a `Simulator` instance.
All public access is through seven module-level functions. **No Gymnasium dependency** in
`env_simulation.py` — students build their own `gymnasium.Env` on top.

**Each team controls a single car (id=0).** Adversary slots (id=1..N-1) exist for
adversarial training. During competition the tournament script injects other teams' policies.

> **Note on adversarial training (open-ended):** The training loop shown below is the
> *minimal example*. How students structure adversarial training — self-play, frozen
> opponent snapshots, training multiple policies separately then fusing/ensembling them,
> curriculum of increasing difficulty — is deliberately left open. It is part of what they
> must design and justify in defense. Do not prescribe a single approach.

### Project phases

| Phase | N cars | Context |
|---|---|---|
| Phase 1 — Solo | 1 | Pure training, no adversaries |
| Phase 2 — Adversarial | up to 4 | Grand Clash, multi-car training |
| Final competition | up to 4 | Live tournament, 1v1v1v1, one camera per car |

**Hard cap: 4 cars maximum on track.** This enables a Mario-Kart-style split-screen
visualization (one follow-camera per car) for the competition. `_MAX_SLOTS = 4`.

---

## Public API (`env_simulation.py`)

```python
def get_space_info() -> dict                 # bounds and metadata, call once at startup
def reset(num_cars: int = 1) -> None          # resets the simulation; lazily rebuilds for N
def get_obs(agent_id: int) -> dict           # full state for one agent
def apply_action(agent_id: int, target_speed: float, steering: float) -> None
def simulation_step() -> None                # advances physics by one decision step
def get_step_info() -> dict                  # global post-step info (ranks, collisions...)
def close() -> None                          # destroys the simulation
```

There is **no `get_my_id`** and there never will be: during training the agent *is* the
controller of whatever slot it drives (it passes the id to `apply_action`); during the
tournament the instructor-side harness decides which slot is which team and only ever hands
each agent its own ego obs. The id is already present as `obs["agent_id"]`. A "tell me who I
am" function makes sense in neither world.

### How students use this — their `env.py` (gymnasium.Env wrapper)

Students wrap these functions inside their own Gym environment. This is where they
compute reward, decide termination, and select/normalize observations:

```python
import gymnasium as gym
from env_simulation import get_obs, apply_action, simulation_step, get_step_info, reset

class RaceEnv(gym.Env):
    def __init__(self, step_limit=8000):
        self.step_limit = step_limit
        # define observation_space / action_space from get_space_info()

    def reset(self, seed=None, options=None):
        reset()                              # num_cars defaults to 1 for solo training
        obs = self._preprocess(get_obs(0))   # their feature selection + normalization
        return obs, {}

    def step(self, action):
        target_speed, steering = self._denormalize(action)   # their action mapping
        apply_action(0, target_speed, steering)
        simulation_step()                                  # advance physics ONCE
        info = get_step_info()
        raw = get_obs(0)
        obs = self._preprocess(raw)
        reward = self._compute_reward(raw, info)          # their reward design
        terminated = info["lap_complete"][0]               # clean episode end
        truncated  = info["step_count"] >= self.step_limit # or info["stagnation"][0]
        return obs, reward, terminated, truncated, info
```

The trained network is exported to ONNX. For competition, `agent.py` loads the ONNX
model and reproduces the same preprocessing/postprocessing inside `predict()`.

> **The observation contract is the dict, not the network tensor.** `env_simulation`
> returns the full obs dict (fixed 100-ray scan + scalars + opponents) and the full
> `get_step_info`. Each team selects/reduces whatever they want **inside their own
> `agent.py` preprocessing**, producing their own input tensor. Two agents with completely
> different network input shapes therefore interoperate in the same tournament — the harness
> only ever sees the uniform `(obs_dict, info) -> (speed, steer)` interface. The ONNX is
> encapsulated behind `agent.py`.

### Minimal adversarial loop (example only — not prescriptive)

```python
reset(num_cars=4)
while True:
    obs = get_obs(0)
    apply_action(0, *my_policy(obs))
    for agent_id in [1, 2, 3]:               # up to 3 adversaries (4 cars total max)
        apply_action(agent_id, *some_opponent_policy(get_obs(agent_id)))
    simulation_step()                         # EXACTLY ONCE per loop
    info = get_step_info()
    # reward + termination = student's responsibility
    if my_termination_condition(obs, info):
        break
close()
```

**Critical:** `apply_action()` only registers the action. `simulation_step()` advances
physics — call exactly once per loop, after all `apply_action` calls.

---

## Function reference

### `get_space_info() -> dict`

```python
{
    "observations": {
        "lidar":      {"shape": (N_RAYS,), "bounds": (0.0, 15.0),   "unit": "meters"},
        "velocity":   {"shape": "scalar",  "bounds": (-5.0, 5.0),   "unit": "m/s"},
        "steering":   {"shape": "scalar",  "bounds": (-0.4189, 0.4189), "unit": "rad"},
        "progress":   {"shape": "scalar",  "bounds": (0.0, 1.0),    "unit": "normalized lap"},
        "lap_count":  {"shape": "scalar",  "bounds": (0, inf),      "unit": "int"},
        "opponents":  {"shape": "dict[0..3]",                       "unit": "see below"},
    },
    "actions": {
        "target_speed": {"bounds": (-0.6, 5.0),  "unit": "m/s target velocity (asymmetric: fwd 5.0, rev 0.6)"},
        "steering":     {"bounds": (-0.4189, 0.4189), "unit": "rad"},
    },
    "decision_freq_hz": 20,   # control rate; one simulation_step() = one decision (see below)
}
```

`N_RAYS` reflects the current LiDAR ray count constant (`_LIDAR_RAYS`, default 100; see LiDAR section).

**Friction is NOT listed here** — it is not an observation. It is accessible in
`get_step_info()` for students who discover it and choose to use it (a smart bonus,
intentionally undocumented to students).

**Scalars**: `velocity`, `steering`, `progress`, `lap_count` are Python `float`/`int`
scalars, NOT `np.ndarray` shape `(1,)`. Students wrap in `np.array([v])` themselves
if needed for tensor concatenation. (Fixes review gap 1.)

**Velocity bounds** `(-5.0, 5.0)` are *observation* bounds (measured body-frame speed, which
can briefly go negative in collisions). They are NOT the action bounds — the *commanded*
target speed is asymmetric `(-_REV_MAX_MS, _FWD_MAX_MS) = (-0.6, 5.0)`. (Fixes review gap 2.)

**Steering bound** `0.4189 rad` matches the engine's `s_min / s_max` exactly; the wrapper
clamps with `np.clip(steering, _PARAMS["s_min"], _PARAMS["s_max"])`, and the engine enforces
its own limits underneath.

> The space contract is **unchanged** by the per-car odometer lap detection: `progress`
> stays a scalar in `[0,1)` (now the *current car's lap phase measured from its own start*),
> `lap_count` stays an int starting at 0. See Track & centerline.

### `reset(num_cars: int = 1) -> None`

Resets simulation. `num_cars ∈ [1, _MAX_SLOTS]`.

**Lazy rebuild:** if `num_cars` differs from the current `Simulator`'s agent count, the
underlying `Simulator` is rebuilt with the new count; otherwise it is reused and agents are
just repositioned to their start poses. (f1tenth's `num_agents` is a *constructor* parameter
of `Simulator`, not a `reset` argument — the lazy rebuild hides this behind our reset.)

Friction starts at maximum (1.0). All per-agent state cleared: steering history, progress,
**per-car odometer (`cum`, `rel_prev`, `progress_start`)**, lap counts (back to 0), DNF flags.
`progress_start[i]` is recomputed from each car's freshly-placed start pose (see Start grid).

### `set_map(name: str, num_cars: int | None = None) -> None`

Load a map circuit. Call once before training starts or mid-episode to switch maps.
Idempotent when called with the same map name as currently loaded.

**Parameters:**

- `name`: circuit name (e.g., `"example"`, `"bahrain"`, `"monaco"`). Valid names returned by
  `get_available_maps()`. Maps are loaded from the `maps/` directory; each must have a
  `<name>.csv` centerline, optionally a `<name>.png` occupancy grid and `<name>.yaml` config.
- `num_cars`: optional — if different from current `_num_agents`, triggers lazy rebuild with
  the new count (same logic as `reset(num_cars=N)`).

**Map loading priority:** centerline CSV > occupancy grid PNG > map YAML. If no centerline CSV
exists, a fallback start pose is derived from the first waypoint of the CSV or the map YAML
`origin` (default yaw = `_MAP_STHETA = 0.0`). A `RuntimeWarning` is raised if fallback is used.

Maps must have distinct names — duplicate circuit names raise a `ValueError`. After calling
`set_map()`, cars are placed on the 2×2 grid for the loaded map and per-car start positions
are recomputed accordingly (each map has different centerline geometry, so start poses differ).

### `get_available_maps() -> list[str]`

Return a sorted list of available circuit names. Each name corresponds to a directory in
`maps/` containing a `<name>.csv` centerline file and optionally `<name>.png` / `<name>.yaml`.
Use this to discover available circuits without hardcoding map names:

```python
from env_simulation import get_available_maps, set_map

available = get_available_maps()
print(f"Available circuits: {available}")
set_map(available[0])  # pick the first available circuit
```

### `get_start_pose(slot_index: int) -> tuple[float, float, float]`

Return `(x, y, yaw)` for a 2×2 grid slot index (`0..3`). Reads from `_start_poses_for_map`
(current map's start poses). Raises `ValueError` if called before `set_map()` or with an
invalid slot index.

Useful for verifying the 2×2 layout or placing agents at specific positions:

```python
from env_simulation import get_start_pose, get_space_info

bounds = get_space_info()["actions"]["target_speed"]["bounds"]
print(f"Car 0 (front-left):  x={x:.1f} y={y:.1f} yaw={yaw:.2f}")
print(f"Car 1 (front-right): x={x:.1f} y={y:.1f} yaw={yaw:.2f}")
print(f"Car 2 (back-left):   x={x:.1f} y={y:.1f} yaw={yaw:.2f}")
print(f"Car 3 (back-right):  x={x:.1f} y={y:.1f} yaw={yaw:.2f}")
```

On the **example** map the start grid is:

| Slot | Position          | Description          |
|------|-------------------|----------------------|
| 0    | front-left        | Front, left of centerline |
| 1    | front-right       | Front, right of centerline |
| 2    | back-left         | Behind, left of centerline |
| 3    | back-right        | Behind, right of centerline |

> **Note:** Cars start with `velocity=0` and `steering=0` after reset. The yaw values are all
> identical (aligned to the centerline tangent at the start line). The lateral/longitudinal
> offsets are computed from `_GRID_LAT_M` and `_GRID_ROW_M` relative to the centerline point.

### `get_obs(agent_id: int) -> dict`

```python
{
    "agent_id":  int,          # id passed in
    "lidar":     np.ndarray,   # (N_RAYS,) meters, transparent to adversary vehicles
    "velocity":  float,        # m/s, body-frame forward, signed
    "steering":  float,        # rad — last *commanded* value, NOT a joint sensor
                               # 0.0 if apply_action not yet called this episode
    "progress":  float,        # [0.0, 1.0), lap phase from THIS car's start; resets to 0
                               #   at the car's own start point each lap
    "lap_count": int,          # starts at 0, latched floor(cum), increments at each lap
    "rank":      int,          # race position (1 = first), by cumulative distance (cum)
    "opponents": dict,         # see below
}
```

`progress` is **computed by the wrapper** by projecting the car pose onto the track
centerline and re-referencing it to the car's own start (see Track & centerline). The engine
itself only exposes raw poses and a finish-line lap counter; the dense per-car `[0,1)` phase
and the lap odometer are ours.

### Opponents — egocentric body frame

Opponents are expressed in the **ego body frame**, not absolute world coordinates.
This prevents geometric controllers (Pure Pursuit, MPC on reconstructed map) from being
dropped in as RL replacements — the global map cannot be reconstructed from this data.

For opponent at world $(X_i, Y_i, \psi_i)$, ego at $(X_0, Y_0, \psi_0)$:

$$\begin{bmatrix} x_\text{rel} \\ y_\text{rel} \end{bmatrix} =
\begin{bmatrix} \cos\psi_0 & \sin\psi_0 \\ -\sin\psi_0 & \cos\psi_0 \end{bmatrix}
\begin{bmatrix} X_i - X_0 \\ Y_i - Y_0 \end{bmatrix}$$

$$\psi_\text{rel} = \operatorname{atan2}(\sin(\psi_i - \psi_0),\, \cos(\psi_i - \psi_0))$$

```python
"opponents": {
    0: {
        "x_rel":     float,  # m, ego frame (forward = +x)
        "y_rel":     float,  # m, ego frame (left = +y)
        "yaw_rel":   float,  # rad, relative heading
        "speed":     float,  # m/s scalar
        "progress":  float,  # [0.0, 1.0) that opponent's own-start lap phase
        "lap_count": int,    # that opponent's lap count
        "status":    int,    # 0=INACTIVE/DNF, 1=ACTIVE, 2=FINISHED
    },
    1: { ... },
    2: { ... },
    3: { ... },   # up to slot 3 (4 cars total)
}
```

Inactive slots: `status=0`, all numeric fields `0.0`.
Fixed-key dict preserves readability for preprocessing while guaranteeing fixed shape
for PyTorch/ONNX. Index by slot number, not by live position.

> Opponent `progress` is that opponent's own-start phase, so it is NOT directly comparable
> across cars for "who is ahead" (different start references). Cross-car race position is
> provided by `rank` (computed from the comparable cumulative-distance odometer). This is
> intentional: it keeps opponent data egocentric and non-reconstructible.

### `apply_action(agent_id, target_speed, steering) -> None`

- `target_speed`: target velocity m/s, asymmetric `[-0.6, 5.0]` (forward `_FWD_MAX_MS=5.0`,
  reverse `_REV_MAX_MS=0.6`), silently clamped. The engine's own `v_max` default is 20 m/s —
  we clamp to our bounds ourselves before passing to the engine.
- `steering`: wheel angle rad, `[-0.42, 0.42]`, silently clamped.

**Engine input format:** f1tenth's `Simulator` step takes a per-agent row `[steering_angle, speed]`
(steering first, speed second). `apply_action` registers `(target_speed, steering)`; the wrapper
assembles the `[steering, target_speed]` row at `simulation_step()`.

### `simulation_step() -> None`

One **decision step**: advances the f1tenth single-track dynamic model (RK4 integrator) by
`_PHYSICS_STEPS = 5` sub-steps of `1/_FREQ_HZ` (5 × 10 ms = **50 ms**), so control runs at
`_DECISION_FREQ_HZ = 20 Hz` over 100 Hz physics. The action is held constant across the 5
sub-steps (built-in action-repeat / frame-skip). Call exactly once per loop.
**Frame-skip is fixed, not a student knob** — it keeps the control rate uniform and the step
budget low on weak hardware (see note below).

### `get_step_info() -> dict`

This is the global per-step information students read to compute reward and decide
episode termination. **Used every step** inside their `env.py`.

```python
{
    # No "done" field — termination is the student's responsibility.
    "step_count":       int,            # simulation_step() calls since reset
    "time_elapsed":     float,          # simulated seconds (step_count / _DECISION_FREQ_HZ)
    "ranks":            dict[int, int], # {agent_id: rank} (1 = first), by cum distance
    "collisions": {
        "wall":         dict[int, bool],
        "vehicle":      dict[int, bool],
    },
    "progress_delta":   dict[int, float], # signed wrap-aware odometer delta this step
    "lap_complete":     dict[int, bool],  # True the step the latched lap_count increments
    "stagnation":       dict[int, bool],  # True if agent hasn't progressed in _DNF_WINDOW_SEC
    "friction_current": float,            # current friction (undocumented bonus for students)
    "opponents_mask":   np.ndarray,       # (4,) bool, True = active slot
    "agent_status":     dict[int, int],   # {agent_id: status} 0=DNF, 1=ACTIVE, 2=FINISHED
    "lap_times":        dict[int, list],  # per-agent completed-lap times (s)
    "race_over":        bool,             # True when no slot is still ACTIVE
}
```

`progress_delta[i]` is the signed, wrap-corrected per-step advance fed into the odometer
(positive forward, negative when pushed back), already `±1`-corrected at the wrap. It is the
true distance-fraction covered this step, robust to the `0.99 → 0.01` wrap.

**Termination pattern for students:**
```python
terminated = info["lap_complete"][0]          # crossed own start (a lap booked)
truncated  = info["step_count"] >= my_limit   # or info["stagnation"][0]
```

**Why no `done`**: f1tenth's shell `done` flips on collision OR finishing 2 laps, evaluated
from the ego only (asymmetric, undocumented). By bypassing the shell and delegating to
students we align `terminated`/`truncated` with correct Bellman bootstrapping for your chosen RL algorithm.
Misunderstanding shows up in their training curves — intended pedagogical
moment.

### `close() -> None`

Destroys the underlying `Simulator` and frees engine state. Always call at script end.

---

## Track & centerline

Maps follow the standard ROS occupancy-grid format, exactly as f1tenth ships them:

- **Occupancy grid**: `<map>.png` (or `.pgm`) + `<map>.yaml` with `image`, `resolution`,
  `origin`, `occupied_thresh`, `free_thresh`, `negate`. Consumed by the engine's
  `ScanSimulator2D` for the LiDAR. Example: `example_map.png` / `example_map.yaml`.
- **Centerline / raceline CSV**: required for dense progress. Format per
  `config_example_map.yaml`: delimiter `;`, `rowskip` header lines, column indices
  `xind/yind/thind/vind`. f1tenth_racetracks provides ready-made CSVs for 20+ real circuits.

**Dense progress** `progress ∈ [0,1)` is computed by projecting the car pose onto the
centerline polyline (cumulative arc length), reusing the njit `nearest_point_on_trajectory`
helper from f1tenth's `waypoint_follow.py`. This raw projected progress is in **absolute
track coordinates** (same reference for all cars).

### Lap detection — per-car odometer (finish = each car's own start point)

Lap counting does **not** use a fixed finish line. Each car's lap is measured from **its own
grid start**, so the staggered 2×2 start grid gives every car the exact same distance to
cover — start-position fairness is built into the lap logic itself, not bolted on.

Per car `i`, at `reset()` the wrapper records `progress_start[i]` = the absolute projected
progress of that car's start pose. Each decision step:

```
rel_i   = (progress_abs_i - progress_start[i]) mod 1.0     # 0 at own start, lap phase
d       = rel_i - rel_prev_i
if d < -0.5: d += 1.0          # forward wrap  (0.99 -> 0.01)
elif d > 0.5: d -= 1.0         # backward wrap (0.01 -> 0.99)
cum_i  += d                    # cumulative fractional laps from own start
lap_count_i = max(lap_count_i, floor(cum_i))   # latched, non-decreasing
rel_prev_i = rel_i
```

- `obs["progress"]` = `rel_i` (resets to 0 at the car's own start point each lap).
- `obs["lap_count"]` = the latched count; `lap_complete` fires the step it increments.
- **FINISH** (`status=2`) when `cum_i >= _REQUIRED_LAPS`.
- **Ranking** among ACTIVE cars is by `cum_i` (highest = furthest toward its own finish).
  `cum` collapses `(lap_count, rel)` into a single comparable scalar and is start-grid-fair.

Why this beats a geometric finish-line crossing test:

- **Anti-rebound is free.** Signed accumulation means a car oscillating on its start point
  adds then subtracts the same `d` (net 0) and never books a phantom lap. A finish-line
  segment test would need an explicit forward-direction check *plus* a debounce window.
- **Latched `lap_count`** (`max(lap_count, floor(cum))`) is wobble-proof at integer
  boundaries: projection noise nudging `cum` back and forth across an integer cannot
  double-count.
- **Distance-fair.** Measuring from each car's own start makes the small longitudinal stagger
  of the 2×2 grid carry zero advantage — every car must cover the same `_REQUIRED_LAPS`.
- Verified robust against: mid-track start, 10%-of-steps backward jitter, ±0.2%-lap
  projection noise, on-line oscillation (0 phantom laps), and 1000-lap drift (no accumulated
  error).

> **Caveat (not a regression):** like *any* centerline-projection scheme — including the
> previous `prev < 0.5 → curr ≥ 0.5` crossing logic — a track whose centerline passes within
> ~1 car width of itself (figure-8, very tight switchbacks) can make the projection jump
> between branches and produce a spurious large `d`. Standard f1tenth closed-loop circuits and
> the example map are unaffected. If such a map is ever used, sanitize/segment the centerline.

`(lap_count, progress)` still fully encodes the car's own race state; cross-car position is
`rank`.

### Start grid (2×2)

Cars start on a **staggered 2×2 grid** (two side-by-side rows), not single file. This needs
far less track width than 4 abreast while keeping the per-car odometer distance identical for
every car. For start index `k = 0..3`: row `r = k // 2`, column `c = k % 2`.

```
lateral offset      = (c - 0.5) * _GRID_LAT_M        # perpendicular to start heading
longitudinal offset = -r * _GRID_ROW_M               # behind, along -heading
pose = start_xy
       + lateral_offset      * (-sin θ,  cos θ)       # left/right of the racing line
       + longitudinal_offset * ( cos θ,  sin θ)       # backwards down the line
heading = θ (= _MAP_STHETA) for all cars
```

- Assumes a start-line free width of ≳ `_GRID_LAT_M + car width` (≈ 0.9 m at defaults). On
  narrower starts, reduce `_GRID_LAT_M`.
- **Optional robustness (nice-to-have, not blocking):** probe the occupancy grid at each
  candidate lateral pose and shrink `_GRID_LAT_M` automatically if a cell is occupied,
  falling back toward single-file only as a last resort.
- Because laps are measured per-car from these poses, the longitudinal stagger between the two
  rows introduces **no** distance advantage.

> Design tension to keep in mind: a tight 2×2 grid + contact-allowed racing makes a turn-1
> scramble likely. That is on-brand for Crash&Learn; if a specific demo needs calmer starts,
> widen `_GRID_ROW_M`.

**Custom maps** need a centerline CSV in addition to the occupancy grid. For a map without
one, extract a centerline from the occupancy grid (image skeletonize + smoothing) — a small
one-off script, run once per map.

---

## LiDAR

- **Fixed ray count**: `_LIDAR_RAYS = 100`, set at `Simulator` construction (`num_beams`).
  Uniform beams over the configured field of view. The 100-ray scan is the **shared contract**;
  students decimate/average/sector-select it in their own `agent.py` preprocessing — the
  pedagogical lesson ("feeding the whole raw scan to the net is wasteful") is learned there,
  on a fixed shared input, without breaking interoperability.
- The engine's raw default is 1080 beams / 270°; we set 100 to keep the raycast cost bounded
  on student CPUs (cost scales ~linearly with `num_beams` in the numba scan).
- **Transparent to adversary vehicles** — wall/static obstacles only. Opponent awareness comes
  from `obs["opponents"]`, a separate channel.
- Wrapper-added multiplicative uniform noise ±3% per ray. Range clipped to [0.1, 15.0] m.

> **Performance vs ray count** — the scan is a numba `@njit` raycast (NOT PyBullet
> `rayTestBatch` anymore), and `num_beams` is a constructor parameter. Cost scales ~linearly:
> 1080 beams ≈ 10× the cost of 100. Default 100 is the chosen balance; retune via `_LIDAR_RAYS`
> only if profiling on target student laptops calls for it. Below ~60 beams, thin obstacles and
> tight chicanes start being missed between rays.

---

## Dynamic friction

- At `reset()`: friction = 1.0 (max grip).
- Every random interval in `_FRICTION_INTERVAL_SEC` (seconds): changes to a random value in
  `[_FRICTION_LO, _FRICTION_HI]`, applied via the single-track model's surface friction
  coefficient `mu` (engine `update_params`).
- **`_FRICTION_LO` kept high (≈ 0.75)** — only a mild reduction. Enough to add robustness
  pressure and reward domain randomization, but NOT enough to dominate the
  non-stationarity / partial-observability challenge. Friction is a secondary difficulty,
  not the main one.
- Accessible in `get_step_info()["friction_current"]`, not in `get_obs()`. Undocumented to
  students — a smart bonus if they discover and exploit it.

---

## Collision policy — the Crash&Learn differentiator

This is the deliberate divergence from vanilla f1tenth (clean racing → contact racing).

- **Collisions are never a penalty and never terminate the episode.** `get_step_info`
  exposes `collisions.wall[id]` and `collisions.vehicle[id]` as flags only. The student
  decides whether to avoid or exploit contact. This is free because we bypass
  `F110Env._check_done` — there is no done-on-collision to suppress.
- **No reward penalty is injected by the sim** for either wall or vehicle contact.
- **Vehicle-vehicle contact stays physical** (demolition-derby): cars actually collide.
  The wrapper adds a parameterizable **2D knockback** (equal-mass impulse exchange,
  restitution `_KNOCKBACK_REST`) post-step so contact is playful rather than a dead stop.
  Slight inter-penetration / pile-ups are acceptable features.

### Wall response — full-pose rollback (yaw-clobber fix)

The engine's `check_ttc()` on wall contact does `self.state[3:] = 0`, which zeroes
velocity (index 3) **and yaw_angle (index 4)** and yaw_rate (5) and slip (6).
The actual state layout is `[x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]`
(the `RaceCar` class docstring's `[x,y,theta,vel,...]` ordering is wrong — `steer`
is at index 2, `yaw` at index 4).

Clobbering `yaw` snaps the heading to 0 rad (+x). With sustained forward throttle,
the car re-accelerates along the corrupted heading on the next decision step, escapes
laterally out of the wall into open track, and eventually DNFs off-map at (10000,10000).
This was a latent engine bug: `F110Env._check_done` previously terminated the episode
on first wall contact, masking it entirely. Removing `done`-on-collision exposed it.

**The wrapper fix (in `simulation_step()`):** before each physics sub-step, snapshot
`(state[0], state[1], state[4])` = (x, y, yaw) for every active agent. After the step,
if `agent.in_collision` (wall only in this fork): restore all three from the snapshot.
Velocity/yaw_rate/slip remain at 0 (correct for a stopped car). `ctrl[i,1]` is set to
0.0 for the remaining sub-steps of that decision step.

Consequences:
- A car floored head-on into a wall is **pinned** (yaw preserved pointing into wall,
  v=0, re-acceleration hits wall again each sub-step). DNF by stagnation follows after
  `_DNF_WINDOW_SEC` seconds. Escape requires reversing.
- Glancing/tangential contact: re-acceleration is mostly tangential, clears within one
  sub-step, car is not permanently pinned.
- No speed debuff, no reward penalty, no episode termination (unchanged).
- `agent.in_collision` is wall-only in this fork (NOT OR'd with vehicle collision).
  `collision_idx >= 0` is the vehicle-only source. Do not conflate them.

---

## DNF mechanism

A vehicle is marked `DNF` (status=0) if its **forward progress** does not strictly increase
over a rolling window of `_DNF_WINDOW_SEC` seconds. Forward progress is measured as the dot
product of the car's velocity vector with the track centerline tangent at its nearest waypoint,
making the metric track-aware and robust to circular motion.

### Detection logic

Each decision step for an active agent `i`:

1. Project car pose onto centerline waypoint segment → get local tangent direction `d_i`
2. Compute forward progress: `progress[i] = dot(velocity, d_i)` (signed scalar along track)
3. Maintain running max within current lap: `_max_progress[i] = max(progress[i], _max_progress[i])`
4. Store in rolling window: `_progress_history[i].append(_max_progress[i])`
5. DNF triggers when `max_progress <= window_start`:

```python
_max_progress[i] = max(float(_progress[i]), _max_progress[i])
_progress_history[i].append(_max_progress[i])

window_start = _progress_history[i][0]
_stagnation_flag[i] = (_max_progress[i] <= window_start)  # strictly must increase
```

On DNF (engine-state based):
1. Move the car out of the scene (off-map pose) so it stops interfering with others' LiDAR.
2. Freeze its state (zero velocity, ignore further actions for that slot).
3. `opponents_mask` slot → `False`, `agent_status` → `0`.

### Key properties

- **Track-aware**: Uses waypoint-projected forward progress, not raw delta (which could be 0 in pure circular motion even while moving).
- **Reset-on-lap**: `_max_progress[i] = 0.0` at lap completion, so the metric works per-lap rather than cumulatively.
- **No threshold tuning**: Strictly-increasing check (`>` comparison) is unambiguous — any forward movement eventually breaks a stuck state.

Handles stuck agents in competition without corrupting the simulation state.

---

## Vehicle lifecycle — status values

| Value | State | Description |
|---|---|---|
| `0` | `INACTIVE` / `DNF` | Moved off-map, frozen |
| `1` | `ACTIVE` | Racing |
| `2` | `FINISHED` | Completed `_REQUIRED_LAPS`, removed from scene |

`lap_count` starts at 0 (latched `floor(cum)`), increments as the car re-crosses its own start.
`progress` (= `rel`) resets to 0.0 at the car's own start point each lap. `cum` (internal)
encodes cross-car race position and drives `rank`.

---

## Hidden parameters (not distributed)

All physics constants are **hardcoded in `env_simulation.py`** as module-level constants.
`config/sim_config.yml` is **documentation only — not loaded at runtime**. Editing it has
zero effect.

**All time-based parameters are expressed in SECONDS in the code** and converted to *decision
steps* at runtime via `_DECISION_FREQ_HZ` (bookkeeping and DNF run once per `simulation_step`,
i.e. at 20 Hz — NOT at the 100 Hz physics rate). Converting via `_FREQ_HZ` would make those
timers 5× too long.

| Constant | Value | Unit | Note |
|---|---|---|---|
| `_FREQ_HZ` | 100 | Hz | physics rate, 10 ms timestep (single-track RK4) |
| `_DECISION_FREQ_HZ` | 20 | Hz | control rate; one `simulation_step()` = 1 decision |
| `_PHYSICS_STEPS` | 5 | — | physics sub-steps per decision (`_FREQ_HZ / _DECISION_FREQ_HZ`); action held constant (frame-skip) |
| `_FWD_MAX_MS` | 5.0 | m/s | forward target-speed cap |
| `_REV_MAX_MS` | 0.6 | m/s | reverse target-speed cap (asymmetric) |
| `_FRICTION_LO` | 0.75 | — | mild reduction only (engine `mu` floor) |
| `_FRICTION_HI` | 1.0 | — | upper bound, also reset value |
| `_FRICTION_INTERVAL_SEC` | [30, 45] | s | random interval between changes |
| `_DNF_WINDOW_SEC` | 8.0 | s | stagnation window (converted: 160 decisions at 20 Hz) |
| `_LIDAR_RAYS` | 100 | — | total rays (uniform); engine `num_beams` |
| `_MAX_SLOTS` | 4 | — | hard cap, 4 cars max |
| `_REQUIRED_LAPS` | 3 | — | laps to FINISH (status=2); finish at `cum >= 3` |
| `_KNOCKBACK_REST` | 0.5 | — | restitution for vehicle-vehicle impulse |
| `_GRID_LAT_M` | 0.6 | m | lateral gap between the two cars of a row (2×2 grid) |
| `_GRID_ROW_M` | 0.8 | m | longitudinal gap between the two grid rows |

> `_DEBUFF_SPEED_MS` and `_DEBUFF_SEC` are **gone** (wall debuff removed). `_GRID_SPACING`
> from the single-file grid is replaced by `_GRID_LAT_M` + `_GRID_ROW_M`.

## Parameters communicated to students

- `num_cars`: 1 for solo, up to 4 for adversarial
- Friction varies during race (not the bounds, not how to read it)
- Walls stop you on contact (physical; no reward term)
- **Vehicle collisions are NOT penalized and do NOT end the episode** (contact racing)
- Action/observation bounds via `get_space_info()`
- `lap_count` (starts 0) and `progress` reset semantics (`progress` is your own-start lap phase)
- Opponents are in ego body frame (needed to preprocess correctly)
- Control runs at 20 Hz (`decision_freq_hz` in `get_space_info()`); one `simulation_step()` = one decision

---

## Physics frequency

**100 Hz physics (10 ms), 20 Hz control.** Each `simulation_step()` runs 5 single-track-model
sub-steps (action held constant) and exposes state once → students see a 20 Hz MDP over an
100 Hz simulation. The single-track dynamic model (Pacejka-style tire forces, slip) integrated
in the plane gives realistic drift/grip behavior without any 3D rigid-body cost. Inter-vehicle
contact artifacts (slight inter-penetration, pile-ups) are acceptable features of the
demolition-derby design.

> **Note — no student frame-skip knob.** Action-repeat is fixed at `_PHYSICS_STEPS = 5`.
> This was removed deliberately: it keeps the control rate uniform across all submissions
> (fair tournament), keeps the per-second step budget low on weak laptops, and removes a
> confusing hyperparameter. Students still control γ, n-step, etc.; they just don't choose
> the sim/decision ratio.

---

## Expected performance (headless, 100 rays)

The 2D numba engine is substantially faster than the previous PyBullet backend.

**Measured on target Docker image** (`python:3.11-slim`, CPU-only, `crash_learn_base:local`)
— warmup 200 steps, measurement 2 000 steps, single core:

| Config | Decision-steps/s | Real-time multiple |
|---|---|---|
| N=1 (solo training) | **≈ 6 000** | ≈ 300× |
| N=4 (adversarial, 6 GJK pairs) | **≈ 500** | ≈ 25× |
| Vectorized (SubprocVecEnv ×K) | ≈ K × N=1 rate | scales with cores |

> **N=4 note**: the 12× slowdown vs N=1 is dominated by GJK collision detection (O(N²)
> pairs = 6 pairs at N=4 vs 0 at N=1) plus 4 independent LiDAR scans. At 500 steps/s the
> sim still delivers ~30 000 decision-steps/min on a single core — well above the RL data
> rate needed for any RL algorithm. SubprocVecEnv over multiple cores scales the throughput linearly.

### Tournament timing & live play

A tournament decision step (mono-process) is sim (~2 ms at N=4) + 4 ONNX inferences of small
MLPs (~0.5–4 ms total) ≈ **6 ms wall-clock**, against a 50 ms real-time budget per decision.
Compute is therefore **never the live bottleneck** (≈ 8× headroom). Two things, not compute,
are what "live" needs:

1. A **real-time throttle** in `tournament.py` (`--realtime`): `sleep(max(0, 0.05 - elapsed))`
   per decision so the race plays at real speed instead of ~8× too fast.
2. A **real-time renderer**. Matplotlib/Agg is for offline MP4, not smooth live; the pyglet
   `viz-gpu` path (or a light custom renderer) is the live option. The harness compute is fine
   either way.

---

## Dependency / install model

`env_simulation.py` and the simulator install **without `gym` or `gymnasium`**. The f1tenth
engine modules are imported directly (by path / PYTHONPATH); f1tenth's `setup.py` is not used.
Real runtime deps only: `numpy numba scipy Pillow pyyaml`. For inference/tournament,
**`onnxruntime` is pinned in the Docker image** (see Submission). Students write their own
`gymnasium.Env` (~30 lines) for training; the `terminated`/`truncated` distinction is theirs.

> If recon shows the engine requires `numpy<=1.22` (numba-era constraint) on the host Python,
> pin the Docker image to Python 3.10; otherwise modern numpy/numba on 3.11/3.12 is preferred.

---

## Anti-cheat

- Private f1tenth fork: physics params differ from public `f1tenth/f1tenth_gym`
- Opponents in ego frame: geometric controllers can't be dropped in
- Non-default LiDAR (100 beams, configurable)
- Dynamic friction absent from public repo
- Per-car odometer finish (no single fixed line to game) and unknown maps
- No reward function in wrapper
- Agents never import `env_simulation`; `apply_action` on another agent's id is structurally
  out of reach (tournament harness is instructor-side)

---

## Visualization

Default path is **CPU-only matplotlib, headless-capable**. The f1tenth native pyglet/GL
renderer is available as an **opt-in GPU extra**, never on the default import path.

### 1. `viz_topdown.py` — matplotlib 2D top-down (daily use, default)

Reads engine state directly (poses, scans, progress, friction). **Agg backend** for headless
PNG/MP4 (no display, runs in any Docker), **TkAgg** for an interactive window over plain X
(no GL — light, unlike pyglet). Per frame: track boundary from the occupancy grid, ego
rectangle + heading arrow, LiDAR rays (front brighter), adversaries as colored arrows, text
overlay (step, velocity, progress, friction, real_time_ratio, scenario, `--model`).

```bash
# Headless PNG snapshot (no display):
docker compose run --rm viz python viz_topdown.py --no-display --steps 200
# output: ./recordings/viz_<timestamp>.png

# Interactive window via plain X (Linux host, no GPU needed):
xhost +local:docker
docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  viz python viz_topdown.py --label "random agent test"

# Visualize a trained policy instead of random:
docker compose run --rm viz python viz_topdown.py --model submission/model.onnx --no-display
```

### 2. `record_episode.py` — MP4 (headless, matplotlib)

Matplotlib top-down rendering → MP4 via imageio-ffmpeg (bundled encoder, no system ffmpeg,
no PyBullet `getCameraImage`, no GL). 20–30 fps output to `./recordings/episode_<timestamp>.mp4`.

```bash
docker compose run --rm viz python record_episode.py --steps 5000
```

### 3. `viz-gpu` — f1tenth native pyglet renderer (optional, GPU + X)

The original f1tenth renderer (pyglet/OpenGL) for students who have a GPU and a display and
want the nicer visual, and the realistic path for **live** tournament rendering. **Opt-in
only**: a separate compose service with nvidia flags + `DISPLAY` + `/tmp/.X11-unix`. pyglet is
**lazy-imported** (only when this path runs), so CPU-only laptops without X never hit it.

```bash
docker compose up viz-gpu        # requires nvidia-docker + X
```

### GPU policy (Epitech philosophy)

GPU is **optional and allowed**, never required. CPU is the default path for everything.
Students who have a GPU may use it for (a) the nicer pyglet rendering (`viz-gpu`) and (b) RL
training (e.g., with `device="cuda"`). Not using a GPU costs only render
prettiness and training speed, never correctness.

---

## Competition submission format

```
submission/
├── model.onnx
└── agent.py
```

No `requirements.txt`: it was training-only (torch, sb3, …) and has no place in an inference
submission. At inference `agent.py` may import **only** `onnxruntime` and `numpy`. The
`onnxruntime` version is **pinned in the instructor Docker image**; students must export their
ONNX with an **opset ≤ the announced maximum** (the pinned runtime rejects newer IR/opset at
load — and `verify_submission.py` surfaces that on the student's machine, not at the
tournament).

### `agent.py` required interface

```python
import os
import onnxruntime

class Agent:
    def __init__(self):
        # Resolve the model path relative to THIS file, never the cwd:
        # the tournament loads several agents from different folders in one process.
        here = os.path.dirname(os.path.abspath(__file__))
        self.model = onnxruntime.InferenceSession(os.path.join(here, "model.onnx"))

    def predict(self, obs: dict, info: dict) -> tuple[float, float]:
        # obs  = raw get_obs() for this agent's car only
        # info = full get_step_info()
        # 1. Preprocess obs however you trained (feature selection, normalization,
        #    ego-frame opponent handling) into the model's input tensor.
        # 2. Run the ONNX model to get the raw action, then postprocess/denormalize into
        #    physical target_speed and steering.
        # 3. Return (target_speed, steering) as pure Python float scalars — NOT numpy.
        #    np.float32 will be rejected by verify_submission.py. Cast explicitly.
        return float(target_speed), float(steering)
```

### `verify_submission.py` (distributed to students)

Validates a submission before the tournament, using the **same `agent_loader` the tournament
uses** (so "passes verify" means "loads and runs in the tournament"). Modes:

**Type-conformance check (default):** runs 5 blank inference steps, verifies `Agent()`
instantiates, the ONNX loads under the pinned `onnxruntime`, `predict()` returns
`(float, float)` pure scalars, and `agent.py` imports neither `env_simulation` nor anything
outside the allowed set.

**Solo MP4 check (`--record`):** runs the submitted policy solo on the circuit and generates
an MP4 (matplotlib, headless) so the student can see their agent drive before submitting.

**Self-play Grand Prix check (`--grandprix`):** clones the submitted policy onto all 4 cars
and runs a 1v1v1v1 race, generating an MP4 of the full grand prix — this exercises the exact
tournament load+predict path.

```bash
docker compose run --rm sim python verify_submission.py --agent submission/
docker compose run --rm sim python verify_submission.py --agent submission/ --record
docker compose run --rm sim python verify_submission.py --agent submission/ --grandprix
```

### Tournament harness (instructor-side, not distributed)

Lives in a non-distributed `tournament/` package, **mono-process**:

```
tournament/                    # not distributed
  tournament.py                # orchestrator: read bracket.yaml, run heats, aggregate, --realtime
  agent_loader.py              # load + validate ONE submission (SHARED with verify_submission.py)
  scoring.py                   # heat finishing order -> F1 points -> cumulative standings
  bracket.yaml                 # teams + heat format
submissions/
  <team_name>/{model.onnx, agent.py}
recordings/                    # per-heat MP4
results/                       # standings json/csv
```

- **Loading (mono-process):** each `submission/<team>/agent.py` is imported via
  `importlib.util.spec_from_file_location` under a **unique module name** (plain
  `import agent` would only ever load the first). All 4 agents coexist in one interpreter;
  this is why `agent.py` may not pull heavy/conflicting deps (onnxruntime + numpy only) and
  why the model path must be `__file__`-relative.
- **Per-`predict()` timeout:** each call is wrapped in a hard timeout (`signal.alarm` /
  SIGALRM on Linux). Timeout or exception → that slot is DNF'd for the heat and the race
  continues. This catches infinite loops and Python exceptions; a native segfault/OOM in
  onnxruntime would still take down the process (acceptable: disqualify + rerun). Full
  process isolation per agent is the only thing a multi-process design would add, and is
  judged not worth the plumbing here.
- **Heats & fairness:** a heat is up to 4 cars on the 2×2 grid. Because laps are per-car
  odometer (equal distance) the grid is already start-fair; interaction noise (blocking,
  pile-ups) is smoothed by running **N = 2–3 heats with slot permutation** and summing F1
  points. Auto-generation of heats/permutations from a full team list is deferred — start
  with an explicit heat list in `bracket.yaml`.

```python
# tournament.py inner loop (sketch)
from env_simulation import get_obs, get_step_info, apply_action, simulation_step, reset

reset(num_cars=4)
while True:
    info = get_step_info()
    for slot, agent in enumerate(heat_agents):      # heat_agents: slot -> loaded Agent
        t, s = safe_predict(agent, get_obs(slot), info)   # SIGALRM-guarded; DNF on fail
        apply_action(slot, t, s)
    simulation_step()
    if realtime:
        sleep(max(0.0, 0.05 - step_elapsed))
    if info["race_over"]:
        break
```

Agents never import or call `env_simulation`. `apply_action` on another agent's id is
structurally impossible. Only ego obs are passed to each agent. The multi-agent orchestration
(simultaneous stepping, collision aggregation, ranking, lap logic) lives in `_Sim`.

---

## Debugging quick reference

```bash
# Run the full test suite:
docker compose run --rm sim python test_contract.py
docker compose run --rm sim python test_integration.py

# Quick headless smoke test (no display, prints obs/info):
docker compose run --rm sim python demo.py

# Check a single function manually:
docker compose run --rm sim python -c "from env_simulation import get_space_info; print(get_space_info())"

# Clean up orphan containers / networks:
docker compose down --remove-orphans

# Docker permission denied → activate docker group in current shell:
newgrp docker        # or prefix any command with:  sg docker -c "..."
```

---

## Implementation status — f1tenth migration

| # | Item | Status |
|---|---|---|
| A1 | Recon: `base_classes.Simulator` imports without gym on modern numpy/numba | ✅ |
| A2 | Recon: Simulator vs F110Env boundary (multi-agent step, poses, collisions, lap, ranking) | ✅ |
| A3 | Recon: collision physics probe (bounce / stop / interpenetrate) | ✅ |
| B1 | `_Sim` singleton wrapping `Simulator` (not `F110Env`, not `gym.make`) | ✅ |
| B2 | `reset(num_cars=N)` lazy rebuild (1..4) | ✅ |
| B3 | Fixed LiDAR `num_beams=100`, transparent to vehicles, ±3% noise, [0.1,15] clip | ✅ |
| B4 | Centerline CSV → dense `progress ∈ [0,1)` (njit projection) | ✅ |
| B4b | **Per-car odometer lap detection** (own-start finish, signed wrap, latched count, finish at `cum≥_REQUIRED_LAPS`, ranking by `cum`) | ✅ |
| B5 | Dynamic friction via engine `mu` (update_params) | ✅ |
| B6 | Wall response | ✅ (REMOVED — engine iTTC + wrapper full-pose rollback on wall contact; see Wall response section) |
| B7 | DNF (net-displacement window → off-map + freeze + mask) | ✅ |
| B8 | Opponents → egocentric frame, fixed-key dict | ✅ |
| B9 | `get_step_info` contract (no `done`, scalars, masks, status) | ✅ |
| B10 | Collision policy: never penalize, never terminate; 2D knockback | ✅ |
| B11 | Reimplement F110Env-side multi-agent orchestration inside `_Sim` | ✅ |
| B12 | Remove legacy-gym hacks from student `env.py` / `train_ppo.py` | [ ] |
| B13 | **2×2 staggered start grid** (`_GRID_LAT_M` / `_GRID_ROW_M`, record `progress_start[i]`) | ✅ |
| C1 | Slim Dockerfile (no gym, no nvidia base, no mandatory GL/X) | ✅ |
| C1b | **Pin `onnxruntime` in Docker image; announce max opset** | [ ] |
| C2 | `docker-compose.yml`: `sim` / `train` / `tensorboard` / `viz` (headless) | ✅ |
| C3 | `viz-gpu` opt-in service (pyglet, lazy-imported, nvidia flags) | ✅ |
| C4 | `viz_topdown.py` matplotlib (Agg headless / TkAgg) | ✅ |
| C5 | `record_episode.py` matplotlib → MP4 | ✅ |
| C6 | Delete `demo_gui.py` + noVNC/Xvfb/x11vnc plumbing | ✅ (never existed in this branch) |
| D1 | `verify_submission.py` (type check + `--record` + `--grandprix`), uses shared `agent_loader` | [ ] |
| D2 | `tournament.py` instructor harness (mono-process, SIGALRM timeout, `--realtime`, scoring) | [ ] |
| D3 | **`agent_loader.py` shared by verify + tournament** (importlib unique-name load + validation) | [ ] |
| E1 | Remove racecar_gym / dead PyBullet paths | [ ] |

> **Tests must be kept in sync** with the API (no `done`, dict opponents, egocentric frame,
> per-car odometer lap semantics, `progress` = own-start phase, scalar types, `target_speed`
> key, `reset(num_cars=N)`, 20 Hz decision rate, collision-not-terminating). Re-run
> `test_contract.py` and `test_integration.py` after each wrapper change.