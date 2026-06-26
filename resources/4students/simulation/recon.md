# Recon — f1tenth_gym Engine Direct-Drive Investigation
> Go/no-go for SP-B. Probed 2026-06-22.
> Python 3.x · numpy 2.4.6 · numba 0.65.1 · scipy 1.18.0 · venv (no gym, no gymnasium).

---

## Q1 — Import without `gym`, modern numpy/numba

**Result: PASS.** `from f110_gym.envs.base_classes import Simulator` works on numpy 2.4.6 +
numba 0.65.1 with zero numba/numpy incompatibility errors.

**What blocks a clean import and the fix:**
`f110_gym/__init__.py` calls `from gym.envs.registration import register`.
`f110_gym/envs/__init__.py` re-exports `F110Env` from `f110_env.py`, which does
`import gym` and `import pyglet`.
Neither `gym` nor `pyglet` is installed; both must be stubbed with empty modules before the
package is imported. The stubs needed:

```python
import sys, types

def _mk(name):
    m = types.ModuleType(name); m.__path__ = []; return m

for name, mod in [
    ("gym",                   _mk("gym")),
    ("gym.error",             _mk("gym.error")),
    ("gym.spaces",            _mk("gym.spaces")),
    ("gym.utils",             _mk("gym.utils")),
    ("gym.utils.seeding",     _mk("gym.utils.seeding")),
    ("gym.envs",              _mk("gym.envs")),
    ("gym.envs.registration", _mk("gym.envs.registration")),
    ("pyglet",                _mk("pyglet")),
    ("pyglet.gl",             _mk("pyglet.gl")),
]:
    mod.register = lambda *a, **kw: None
    mod.Env = object
    mod.options = {}
    sys.modules[name] = mod

# Then with PYTHONPATH=<repo>/gym:
from f110_gym.envs.base_classes import Simulator, Integrator, RaceCar
from f110_gym.envs.laser_models import ScanSimulator2D
from f110_gym.envs.dynamic_models import vehicle_dynamics_st
from f110_gym.envs.collision_models import get_vertices, collision_multiple
# All PASS
```

Exact output:
```
PASS — from f110_gym.envs.base_classes import Simulator
PASS — ScanSimulator2D, vehicle_dynamics_st, collision_multiple all import OK
```

No numba/numpy incompatibility. A `UserWarning: Chosen integrator is RK4` is printed when
`Integrator.RK4` is selected — cosmetic only, suppress with `warnings.filterwarnings`.

**Pin recommendation:** no Python version pin needed. numpy 2.x + numba 0.65 works on
Python 3.11/3.12. Do NOT pip-install from setup.py (pins `gym==0.19.0` and `numpy<=1.22`
which conflict). Install only: `numpy numba scipy Pillow pyyaml`.

---

## Q2 — Simulator constructor, step format, return value, performance

### Constructor signature

```
Simulator.__init__(self, params, num_agents, seed, time_step=0.01,
                   ego_idx=0, integrator=Integrator.RK4, lidar_dist=0.0)
```

| Parameter | Type | Default | Note |
|---|---|---|---|
| `params` | dict | — | vehicle physics dict (see below) |
| `num_agents` | int | — | positional, required |
| `seed` | int | — | positional, required |
| `time_step` | float | 0.01 s | physics sub-step dt |
| `ego_idx` | int | 0 | which agent is "ego" |
| `integrator` | `Integrator` | `RK4` | `Integrator.RK4` or `Integrator.Euler` |
| `lidar_dist` | float | 0.0 m | longitudinal LiDAR offset from rear axle |

**No `num_beams` / `fov` in Simulator.** These live in `RaceCar.__init__` (defaults: 1080
beams, fov=4.7 rad) but `Simulator` never forwards them. `RaceCar.scan_simulator` is a
class-level singleton created on first `RaceCar.__init__` call. To use 100 beams, override it
**after** Simulator construction:

```python
sim = Simulator(params, num_agents=1, seed=42, time_step=0.01, integrator=Integrator.RK4)
sim.set_map(map_yaml_path, '.png')

# Override to 100 beams:
RaceCar.scan_simulator = ScanSimulator2D(100, 4.7)
RaceCar.scan_simulator.set_map(map_yaml_path, '.png')
# Also recompute class-level beam geometry arrays for 100 beams:
num_beams, fov = 100, 4.7
scan_ang_incr = RaceCar.scan_simulator.get_increment()
RaceCar.cosines       = np.zeros((num_beams,))
RaceCar.scan_angles   = np.zeros((num_beams,))
RaceCar.side_distances = np.zeros((num_beams,))
dist_sides = params['width'] / 2.
dist_fr    = (params['lf'] + params['lr']) / 2.
for i in range(num_beams):
    angle = -fov/2. + i * scan_ang_incr
    RaceCar.scan_angles[i]  = angle
    RaceCar.cosines[i]      = np.cos(angle)
    # side_distances[i] = near-field clipping distance for check_ttc
    ...  # (exact formula in RaceCar.__init__, lines 131-158)
```

**`params` dict** (from `F110Env` defaults):

```python
params = {
    'mu': 1.0489,        # surface friction (update_params to change)
    'C_Sf': 4.718,       # front cornering stiffness
    'C_Sr': 5.4562,      # rear cornering stiffness
    'lf': 0.15875,       # CG to front axle (m)
    'lr': 0.17145,       # CG to rear axle (m)
    'h': 0.074,          # CG height (m)
    'm': 3.74,           # mass (kg)
    'I': 0.04712,        # yaw inertia (kg·m²)
    's_min': -0.4189,    # min steer (rad)
    's_max':  0.4189,    # max steer (rad)
    'sv_min': -3.2,      # min steer velocity
    'sv_max':  3.2,      # max steer velocity
    'v_switch': 7.319,   # velocity at which a_max scales
    'a_max': 9.51,       # max longitudinal accel (m/s²)
    'v_min': -5.0,       # min velocity (m/s)
    'v_max': 20.0,       # max velocity (m/s)
    'width': 0.31,       # car width (m)
    'length': 0.58,      # car length (m)
}
```

### `step()` input format

```python
observations = sim.step(control_inputs)
# control_inputs: np.ndarray(num_agents, 2)
#   col 0: desired steering angle (rad)
#   col 1: desired velocity (m/s)
# Example (1 agent, steer=0, speed=2 m/s):
action = np.array([[0.0, 2.0]])
```

**Confirmed: steering first, velocity second.** The wrapper's `apply_action(agent_id,
target_speed, steering)` assembles `[steering, target_speed]` at step time — correct.

### `step()` return value

```python
observations = {
    'ego_idx':       int,                    # which index is ego
    'scans':         list[np.ndarray],       # per-agent scan, shape (num_beams,)
    'poses_x':       list[float],            # x per agent (m)
    'poses_y':       list[float],            # y per agent (m)
    'poses_theta':   list[float],            # heading per agent (rad)
    'linear_vels_x': list[float],            # forward velocity, state[3] (m/s)
    'linear_vels_y': list[float],            # lateral velocity (all 0.0 — 2D model)
    'ang_vels_z':    list[float],            # yaw rate, state[5] (rad/s)
    'collisions':    np.ndarray(num_agents,) # 1.0 if EITHER wall OR vehicle hit
}
```

`RaceCar.state` internal layout: `[x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]`

### `reset()` input

```python
sim.reset(poses)
# poses: np.ndarray(num_agents, 3)  — each row is [x, y, theta]
```

### `update_params()` — for dynamic friction

```python
sim.update_params(params_dict, agent_idx=-1)
# agent_idx=-1 → all agents; >=0 → single agent
# Change 'mu' in params to update surface friction
```

### Live scan and benchmark

```
scans[0] shape (default): (1080,)        ← default 1080-beam scanner
scans[0] shape (100-beam): (100,)        ← after RaceCar.scan_simulator override
scan range: [0, lidar_max_range] meters
```

**Single-core throughput (measured):**

```
N=1 agent, 100 beams, RK4, example_map, warmup then 1000 steps:
  1000 steps in 0.031 s → 32,564 steps/s
```

At the DESIGN.md setup (4 physics sub-steps per decision, 20 Hz control):
`32,564 / 4 ≈ 8,141 decision-steps/s` on a single CPU core — over 400× real-time.
N=4 will be slower (O(N²) GJK pairs = 6 pairs vs 0), but still expected well above 1000
decision-steps/s. Replace these numbers with measured SP-C profiling on target student laptops.

---

## Q3 — Engine / Shell boundary map

| Feature | Engine (`Simulator`) | Shell (`F110Env` — must reimplement in `_Sim`) |
|---|---|---|
| Simultaneous multi-agent step | **ENGINE** — `Simulator.step()` loops all agents, then runs GJK for all pairs in one call | — |
| Per-agent pose `(x, y, θ)` | **ENGINE** — `obs['poses_x']`, `obs['poses_y']`, `obs['poses_theta']` | — |
| Inter-vehicle collision detection | **ENGINE** — `Simulator.check_collision()` → `collision_multiple()` GJK; `sim.collisions[i]` and `sim.collision_idx[i]` | — |
| Wall/map collision flag | **ENGINE** — `RaceCar.check_ttc()` iTTC; `agent.in_collision` | — |
| Separate wall vs vehicle flag | **MUST REIMPLEMENT** — engine merges both into `obs['collisions']` (see critical note) | Shell: not separated; we read `agent.in_collision` (wall) vs `sim.collision_idx[i]>=0` (vehicle) |
| Lap counting / finish-line | **MUST REIMPLEMENT** — engine has no notion of track topology or laps | Shell: `F110Env._check_done()` toggle_list heuristic, distance to start pose |
| Ranking | **MUST REIMPLEMENT** — engine has no ranking | Shell: `lap_counts` array only; no explicit rank in F110Env either |
| Done-on-collision | **MUST REIMPLEMENT** (intentionally suppressed) | Shell: `F110Env._check_done()` returns `done=True` on ego collision — we bypass entirely |

### Critical: collision flag separation

The engine's `obs['collisions'][i]` is 1 if agent `i` had **either** a wall iTTC hit **or** a
vehicle-vehicle GJK hit. There is no separate field. The `Simulator.step()` execution order is:

1. `Simulator.check_collision()` → writes `self.collisions` (vehicle-vehicle GJK).
2. Per-agent `agent.update_scan()` → calls `check_ttc()` → sets `agent.in_collision` (wall iTTC).
3. `if agent.in_collision: self.collisions[i] = 1.` → merges wall hit into same array.

**To separate them in the wrapper, read after step:**
- **Wall hit**: `sim.agents[i].in_collision` (bool, set by iTTC vs map scan)
- **Vehicle hit**: `sim.collision_idx[i] >= 0` (int, set by GJK; `collision_idx[i] = j` means agent `i` is in contact with agent `j`)

### F110Env shell items that must be reimplemented in `_Sim`

1. **Lap counting / `lap_complete` detection**: project pose onto centerline, detect `0.99→0.01`
   wrap. F110Env's toggle_list heuristic (distance to start pose ≤ 0.5 m) is fragile for
   multi-agent racing and wrong for arbitrary start positions — do not copy it.
2. **Ranking**: derived from `(lap_count, progress)` per agent. Not in engine at all.
3. **Termination logic**: `F110Env._check_done()` returns `done = self.collisions[ego_idx] OR
   toggle_list >= 4`. We bypass this entirely — termination is the student's responsibility.
4. **`lap_times` and `lap_counts` in obs**: F110Env appends these to the obs dict in its
   `step()`. Engine `Simulator.step()` returns neither. The wrapper adds them.

---

## Q4 — Collision physics probe

### Wall collision (iTTC) — behavior: **STOP**

The wall check is inverse-TTC (`check_ttc_jit`): it fires when the predicted
time-to-contact with the nearest wall falls below `ttc_thresh = 0.005 s` (5 ms).

**On trigger:**

```python
# RaceCar.check_ttc(), base_classes.py lines 242-253
if in_collision:
    self.state[3:] = 0.  # zero vel, yaw_rate, slip_angle
    self.accel = 0.0
    self.steer_angle_vel = 0.0
```

Car stops dead. In the next step, the PID controller recomputes acceleration from the
commanded speed and the car re-accelerates from rest.

**Head-on Q1 probe (first wall hit at step 15):**

```
Step        v0        v1        x0        x1   col0   col1
  14     1.379     1.379     0.100     1.900      0      0
  15     1.474     0.000     0.114     1.886      0      1   ← agent 1 wall iTTC fires
  16     1.569     0.048     0.130     1.886      0      0   ← agent 1 re-accelerates
  17     1.664     0.143     0.146     1.887      0      0
```

Agent 1 velocity → 0 immediately. Agent 0 unaffected. `in_collision` flag cleared next step,
car re-accelerates per commanded speed (5 m/s in this test).

### Vehicle-vehicle collision (GJK) — behavior: **INTERPENETRATE (flag only)**

**GJK head-on probe** (agents placed 0.8 m apart in open track area, both at 5 m/s
opposing):

```
Step        v0        v1        x0        x1  wall0  wall1  veh0  veh1
  14     1.379     1.379     8.100     8.700      0      0     0     0
  15     1.474     1.474     8.114     8.686      0      0     1     1   ← GJK fires
  16     1.569     1.569     8.130     8.670      0      0     1     1   ← GJK persists
  17     1.664     1.664     8.146     8.654      0      0     1     1
  18     1.759     1.759     8.163     8.637      0      0     1     1
  19     1.854     1.854     8.181     8.619      0      0     1     1
```

At step 15: x0=8.114, x1=8.686, gap = 8.686 − 8.114 − 0.58 = **−0.008 m** (overlap).

**Velocities continue accelerating unchanged** through the collision. GJK only sets
`sim.collisions[i] = 1` and `sim.collision_idx[i] = j`. No velocity modification, no
position correction, no impulse. Cars physically interpenetrate.

### Summary

| Collision type | Detection | Engine response | Velocity |
|---|---|---|---|
| Wall (map) | iTTC, fires ~5 ms before contact | state[3:] = 0 (stop dead) | **zeroed → re-accel next step** |
| Vehicle-vehicle | GJK polygon overlap | flag only | **unchanged — interpenetration** |

### Implication for wrapper

**Wall collision**: engine already provides a STOP mechanic. The wrapper wall-debuff
(`_DEBUFF_SPEED_MS` cap on commanded speed for `_DEBUFF_SEC`) is applied on top — the
debuff's purpose is to prevent immediate re-acceleration to full speed.

**Vehicle-vehicle collision**: the engine provides detection (`collision_idx`) but no
physical response. The wrapper **must implement 2D knockback** (impulse exchange, optional
restitution coefficient) applied post-`Simulator.step()` when `collision_idx[i] >= 0` is
detected. Without knockback, cars pile up silently in the same cell, which reads as a
simulation glitch rather than a demolition-derby event.

---

## Go/No-Go verdict: **GO**

| Criterion | Status |
|---|---|
| Import on modern numpy/numba without gym | ✅ PASS |
| Simulator constructor fully documented | ✅ PASS |
| Step format confirmed ([steer, speed] per agent) | ✅ PASS |
| Step return fully documented (poses, scans, vels, collision merged) | ✅ PASS |
| 100-beam override path found (class-var post-override) | ✅ PASS |
| Single-core throughput adequate (>8k decision-steps/s) | ✅ PASS |
| Engine vs shell boundary fully mapped | ✅ PASS |
| Collision physics confirmed (stop/flag/interpenetrate) | ✅ PASS |
| Knockback needed for vehicle-vehicle | ✅ CONFIRMED — SP-B item B10 |
| Lap/ranking/done must be fully reimplemented | ✅ CONFIRMED — SP-B items B4, B9, B11 |

SP-B can proceed on all items. The num_beams class-var override (not a Simulator constructor
param) is the main hidden complexity — handle it in `_Sim.__init__` after construction.
