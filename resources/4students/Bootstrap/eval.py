"""Evaluate trained DQN model on CartPole-v1 with rendering."""

import gymnasium as gym
import torch

from model import QNetwork
from utils import set_seed


def main():
    set_seed(42)

    # hyperparams (must match training)
    obs_dim = 4
    action_dim = 2
    hid_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = QNetwork(obs_dim, action_dim, hid_size).to(device)
    model.load_state_dict(torch.load("cartpole_dqn.pt", map_location=device))
    model.eval()

    env = gym.make("CartPole-v1", render_mode="human")

    rewards = []
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        step_idx = 0

        while not (done or truncated):
            action = torch.tensor([obs], dtype=torch.float32).to(device)
            with torch.no_grad():
                q_values = model(action)
            action = q_values.argmax(dim=-1).cpu().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step_idx += 1

        rewards.append(ep_reward)
        print(f"Episode {ep + 1}: reward={ep_reward:.1f}  steps={step_idx}")

    env.close()
    print(f"\nAverage reward over 5 episodes: {sum(rewards)/len(rewards):.2f}")


if __name__ == "__main__":
    main()