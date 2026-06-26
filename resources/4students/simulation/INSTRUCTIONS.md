# Crash&Learn Grand Prix — Usage Instructions

## Quick Start

```bash
# Build the Docker image
docker compose build sim train tensorboard recorder replay

# Verify the simulation works
docker compose run --rm sim
```

---

## Visualization (Recording + Replay)

The simulation uses a **record-then-replay** workflow: `sim_recorder.py` captures episodes at maximum speed, and `viz_replay.py` replays them with interactive controls.

### Record an episode

```bash
# Record a random-drive episode at maximum speed
docker compose run --rm -v .:/app sim python sim_recorder.py --steps 500 --num-cars 1

# Record with an ONNX model policy for car 0
docker compose run --rm -v .:/app sim python sim_recorder.py --steps 200 --model /app/submission/model.onnx --out /app/episode_demo.npz

# Multi-car recording (up to 4 cars)
docker compose run --rm -v .:/app sim python sim_recorder.py --steps 1000 --num-cars 3 --out /app/episode_3cars.npz

# Output: /app/recordings/episode_XXXX.npz (auto-numbered)
```

### Replay recorded episodes (requires display)

```bash
# Allow Docker container to access X server
xhost +local:docker

# Interactive replay with keyboard controls
docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v .:/app replay python viz_replay.py

# Replay a specific episode file with follow mode (camera tracks car 0)
docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v .:/app replay python viz_replay.py --npz /app/episode_demo.npz --follow

# Custom map file
docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v .:/app replay python viz_replay.py --npz /app/episode_demo.npz --map /app/examples/example_map.yaml
```

#### Replay controls

| Key | Action |
|-----|--------|
| ← / → | Step backward/forward 1 frame |
| Shift+← / Shift+→ | Step backward/forward 10 frames |
| Slider | Seek to any step |
| Trajectories checkbox | Toggle full path overlay |

### Episode Recording & Replay

#### Record episodes to NPZ format (new)

```bash
# Record a random-drive episode at maximum speed
docker compose run --rm -v .:/app sim python sim_recorder.py --steps 500 --num-cars 1

# Record with an ONNX model policy for car 0
docker compose run --rm -v .:/app sim python sim_recorder.py --steps 200 --model /app/submission/model.onnx --out /app/episode_demo.npz

# Multi-car recording (up to 4 cars)
docker compose run --rm -v .:/app sim python sim_recorder.py --steps 1000 --num-cars 3 --out /app/episode_3cars.npz

# Record a pure pursuit episode to NPZ (for replay with viz_replay.py)
docker compose run --rm -v .:/app sim python sim_recorder.py --steps 5000 --controller pure_pursuit

# Output directory: /app/recordings/ (auto-numbered episode_XXXX.npz)
```

#### Replay recorded episodes (no sim dependency, requires X11/display)

```bash
# Don't forget
xhost +local:docker

# Replay latest episode from recordings/
docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v .:/app viz python viz_replay.py

# Replay a specific episode file
docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v .:/app viz python viz_replay.py --npz /app/episode_demo.npz

# Follow mode (camera re-centers on car 0 each step)
docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v .:/app viz python viz_replay.py --npz /app/episode_demo.npz --follow

# Custom map file
docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v .:/app viz python viz_replay.py --npz /app/episode_demo.npz --map /app/examples/example_map.yaml
```

**Replay controls:**

| Key | Action |
|-----|--------|
| ← / → | Step backward/forward 1 frame |
| Shift+← / Shift+→ | Step backward/forward 10 frames |
| Slider | Seek to any step |
| Trajectories checkbox | Toggle full path overlay |

**NPZ file contents:** `lidar`, `velocity`, `steering`, `progress`, `lap_count`, `poses`, `actions`, `status`, `friction`, `sim_dt`, `num_cars`

### Record without replay (headless, max speed)

```bash
# Record episode to NPZ at maximum speed (no display needed)
docker compose run --rm -v .:/app sim python sim_recorder.py --steps 500 --num-cars 1
```

---

## Training

### RL Dependencies Installed in Docker Image

The following libraries are pre-installed for reinforcement learning:

| Package | Purpose |
|---|---|
| `gymnasium` | Gym-style environment API (use this, not the legacy `gym`) |
| `torch` | PyTorch for neural network training |
| `tyro` | CLI argument parsing (optional) |
| `stable-baselines3` + `sb3-contrib` | Pre-trained baselines: PPO, SAC, DDPG, TD3, A2C, etc. |

You can install additional packages inside the container with `docker compose run --rm train pip install <package>` if needed.

### Quick Start

```bash
# Build the image first
docker compose build sim train tensorboard

# Train with default settings (4 parallel environments, save freq 500k steps)
docker compose run --rm train

# Override defaults
docker compose run --rm train python test_RL/train_rl.py --num-envs 8 --save-freq 500000
```

### Arguments (passed via docker compose run)

| Argument | Default | Description |
|---|---|---|
| `--num-envs N` | `4` | Number of parallel SubprocVecEnv workers |
| `--save-freq N` | `500000` | Steps interval between checkpoints (total steps) |

### Monitor training with TensorBoard

```bash
# Start TensorBoard server as a detached service (one-time)
docker compose up tensorboard -d

# Point it at the training logs, then open http://localhost:6006
docker compose exec tensorboard tensorboard --logdir /app/test_RL/tb_logs --host 0.0.0.0 --port 6006
```

### Export trained model to ONNX

```bash
# After training, export the latest model
docker compose run --rm train python test_RL/export_onnx.py
```

---

## Demo & Testing

```bash
# Run the built-in demo (smoke test)
docker compose run --rm sim

# Run contract tests
docker compose run --rm sim python test_contract.py

# Run integration tests
docker compose run --rm sim python test_integration.py
```

---

## Switching Map Circuits

The simulation supports running on different tracks. Maps are stored in `maps/<Name>/`.

```python
from env_simulation import set_map, get_available_maps, reset, get_obs, apply_action, simulation_step, get_step_info, close

# List available maps (discovers under maps/<Name>/)
print(get_available_maps())   # e.g. ['Austin', 'BrandsHatch', 'Catalunya', ..., 'Spa']

# Switch to a map (idempotent — no-op if same map already loaded)
set_map("Spa")                # loads maps/Spa/
set_map("example")            # back to the default/example map

# reset() reuses whatever map is active — no per-call map argument needed
reset(num_cars=1)
for _ in range(500):
    obs = get_obs(0)
    apply_action(0, 3.0, 0.0)
    simulation_step()
    info = get_step_info()
    if info["lap_complete"][0]:
        break
close()
```

**Performance:** A map switch triggers a single `Simulator` rebuild (~5–20 ms, paid once). Repeated calls with the same map name are no-op — tracked internally by `_current_map`.

---

## Custom Commands

Run any Python script inside the container:

```bash
# Run a custom script
docker compose run --rm -v .:/app viz python /app/your_script.py

# Run a shell
docker compose run --rm -v .:/app viz bash

# Install packages inside the container
docker compose run --rm sim pip install your-package
```

---

## Arguments Reference

### `sim_recorder.py` (recording)

| Argument | Default | Description |
|---|---|---|
| `--steps N` | 500 | Number of simulation steps to run |
| `--num-cars N` | 1 | Number of cars (1–4) |
| `--model PATH` | "" | ONNX model file for car 0 policy |
| `--controller {random,pure_pursuit,ppo}` | `random` | Controller type |
| `--out PATH` | auto | Output NPZ file path |
| `--recordings-dir PATH` | `recordings/` | Directory for saved episodes |

### `viz_replay.py` (replay)

| Argument | Default | Description |
|---|---|---|
| `--npz PATH` | "" | NPZ episode file to replay |
| `--follow` | false | Enable camera follow mode (tracks car 0) |
| `--map PATH` | "" | Custom map YAML file |
| `--steps N` | auto | Number of steps in episode |
| `--speed-factor FLOAT` | `1.0` | Replay speed multiplier |

---

## Notes

- **X11 forwarding:** Requires `xhost +local:docker` on the host before running viz with DISPLAY.
- **MP4 recording:** Requires `opencv-python` package. Install in container if needed.
- **GPU rendering:** `viz-gpu` service requires nvidia-docker runtime and NVIDIA graphics drivers.
- **Code live-reload:** The `./ → /app` bind-mount means code changes on the host are immediately available in the container without rebuilding.