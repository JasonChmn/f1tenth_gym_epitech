"""Train DQN on CartPole-v1. Main entry point.

Hyperparams (all visible here):
"""

import os

# Must be set BEFORE any torch import to prevent CUDA hanging
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Limit PyTorch to fewer CPU threads to reduce CPU contention
# PyTorch uses MKL/OpenBLAS by default which spawns threads for all 24 cores
# This restricts it to a subset of cores, reducing overhead and heat
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from buffer import ReplayBuffer
from dqn import DQNAgent
from utils import set_seed

# ── Hyperparameters ──────────────────────────────────────────────
GAMMA = 0.99           # discount factor
LR = 1e-3              # learning rate
BATCH_SIZE = 64        # batch size for experience replay
EPS_START = 1.0        # initial epsilon (exploration)
EPS_END = 0.05         # final epsilon
EPS_DECAY = 0.995      # per-episode epsilon decay factor
WARMUP = 1000          # no training until buffer > this size
TARGET_UPDATE = 10     # update target network every N episodes

# ── Environment & logging ───────────────────────────────────────
ENV_ID = "CartPole-v1"
MAX_EPISODES = 1000
MAX_STEPS = 1000
LOG_DIR = "runs/cartpole_dqn"
SEED = 42


def main():
    set_seed(SEED)

    # create environment
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # agent, buffer, logger
    agent = DQNAgent(obs_dim, action_dim, GAMMA, LR, EPS_START, EPS_END, EPS_DECAY)
    replay_buf = ReplayBuffer(capacity=10000)
    writer = SummaryWriter(log_dir=LOG_DIR)

    episode_rewards = []

    print(f"Start training for {MAX_EPISODES} episodes", flush=True)
    for ep in range(1, MAX_EPISODES + 1):
        obs, _ = env.reset()                         # observation (state)
        done = False
        truncated = False
        ep_reward = 0.0
        step_idx = 0

        max_steps_per_episode = 5000  # safety limit to prevent infinite episodes
        while not (done or truncated) and step_idx < max_steps_per_episode:
            # agent picks action using epsilon-greedy policy
            action = agent.select_action(obs, episode=ep)

            # environment executes action -> returns next_state, reward, done
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store transition in buffer
            replay_buf.add(obs, action, reward, next_obs, done)

            obs = next_obs                           # advance state
            ep_reward += reward
            step_idx += 1

            # ── Training step (after warmup) ───────────────────
            if len(replay_buf) > WARMUP:
                b_states, b_actions, b_rewards, b_next_states, b_dones = \
                    replay_buf.sample(BATCH_SIZE)
                loss = agent.update(b_states, b_actions, b_rewards,
                                    b_next_states, b_dones)

        # ── End of episode: log results ────────────────────────
        episode_rewards.append(ep_reward)

        writer.add_scalar("Reward/episode_reward", ep_reward, ep)
        writer.add_scalar("Reward/reward_per_step", ep_reward / max(step_idx, 1), ep)
        writer.add_scalar("Length/episode_length", step_idx, ep)

        eps = agent.eps_end + (agent.eps_start - agent.eps_end) * (agent.eps_decay ** ep)
        writer.add_scalar("Epsilon/epsilon", eps, ep)

        if len(episode_rewards) >= 100:
            avg_r = sum(episode_rewards[-100:]) / 100
            writer.add_scalar("Reward/avg_reward_100", avg_r, ep)

        # hard update target network every TARGET_UPDATE episodes
        if ep % TARGET_UPDATE == 0:
            agent.update_target_network()

        if ep % 10 == 0:
            print(f"Episode {ep:>4d}  reward={ep_reward:>7.1f}  "
                  f"len={step_idx:<4d}  buf={len(replay_buf):>5d}", flush=True)

    # save final model
    torch.save(agent.policy_net.state_dict(), "cartpole_dqn.pt")
    writer.close()
    env.close()
    print(f"Done. Model saved to cartpole_dqn.pt")


if __name__ == "__main__":
    print("Starting train_cartpole.py", flush=True)
    main()
