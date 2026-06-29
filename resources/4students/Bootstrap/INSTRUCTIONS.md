# DQN on CartPole-v1 — Quick Start

## Setup

```bash
# Create virtual environment with uv
uv venv .venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install gymnasium torch tensorboard
```

## Training

Run the training loop:

```bash
python3 train_cartpole.py
```

This trains a DQN agent on CartPole-v1 for 500 episodes. You should see the episode reward climb to ~500+ (the max) within 200–500 episodes, indicating the pole has been balanced consistently.

### Hyperparameters (edit `train_cartpole.py` if needed)

| Param | Value | Description |
|-------|-------|-------------|
| GAMMA | 0.99 | Discount factor for Bellman target |
| LR | 1e-3 | Adam learning rate |
| BATCH_SIZE | 64 | Minibatch size for experience replay |
| EPS_START | 1.0 | Starting epsilon (exploration) |
| EPS_END | 0.05 | Final epsilon |
| EPS_DECAY | 0.995 | Per-episode epsilon decay |
| WARMUP | 1000 | No training until buffer exceeds this |
| TARGET_UPDATE | 10 | Hard update target network every N episodes |

**Note:** `CUDA_VISIBLE_DEVICES=""` forces CPU-only mode. Remove it if you have a GPU with enough VRAM, or set `CUDA_VISIBLE_DEVICES="0"` to use the first GPU.

## Viewing TensorBoard Logs

```bash
tensorboard --logdir runs/cartpole_dqn
```

Then open http://localhost:6006 in your browser. You'll see:

- **Reward/episode_reward** — per-episode total reward (also called "reward per episode")
- **Reward/reward_per_step** — average reward per time step in the episode
- **Reward/avg_reward_100** — 100-episode moving average
- **Length/episode_length** — episodes last (max = 500)
- **Epsilon/epsilon** — exploration rate over time

## Evaluation

Run the agent for 5 episodes with rendering:

```bash
python3 eval.py
```

This loads `cartpole_dqn.pt`, runs 5 untrained episodes with `render_mode='human'`, and prints the average reward.