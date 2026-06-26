#!/usr/bin/env python3
"""Export the latest CleanRL PPO checkpoint to ONNX + agent.py."""

import sys
from pathlib import Path

import numpy as np
if not hasattr(np, "float"):
    np.float = float

import torch
import torch.nn as nn


# ── Paths ──────────────────────────────────────────────────────────────────────

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
MODEL_DIR    = current_dir / "models"
SUBMISSION_DIR = PROJECT_ROOT / "submission"


# ── Agent class (must match train_rl.py) ─────────────────────────────────────

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """CleanRL-style PPO agent matching train_rl.py architecture."""
    def __init__(self, obs_dim: int = 102, action_dim: int = 2):
        super().__init__()
        # Shared encoder: [128, 128]
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        # Actor head (mean)
        self.actor_mean = layer_init(nn.Linear(128, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        # Critic head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 64), std=1.0),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(self.encoder(x))

    def get_action_and_value(self, x, action=None):
        latent = self.encoder(x)
        action_mean = self.actor_mean(latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        from torch.distributions.normal import Normal
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(latent)


# ── Deterministic Actor wrapper for ONNX export ───────────────────────────────

class DeterministicActor(nn.Module):
    """Wrapper that only exports the deterministic actor (no sampling)."""
    def __init__(self, agent):
        super().__init__()
        self.encoder = agent.encoder
        self.actor_mean = agent.actor_mean

    def forward(self, observation):
        latent = self.encoder(observation)
        return self.actor_mean(latent)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export latest PPO checkpoint to ONNX + agent.py")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .pt checkpoint (default: models/racecar_ppo_latest.pt)")
    args = parser.parse_args()

    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        # Prefer latest, fall back to final
        model_path = MODEL_DIR / "racecar_ppo_latest.pt"
        if not model_path.exists():
            model_path = MODEL_DIR / "racecar_ppo_final_model.pt"

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    SUBMISSION_DIR.mkdir(exist_ok=True)

    print(f"Loading {model_path} ...")

    # CleanRL .pt format (state dict only)
    obs_dim = 102  # RaceEnv observation dimension
    agent = Agent(obs_dim=obs_dim)
    state_dict = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    agent.load_state_dict(state_dict)

    agent.eval()

    # ── ONNX export ────────────────────────────────────────────────────────────
    onnx_path = SUBMISSION_DIR / "model.onnx"
    # Batch size 1 as dynamic dimension (standard for inference)
    dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32)

    actor = DeterministicActor(agent)
    actor.eval()
    with torch.no_grad():
        torch.onnx.export(
            actor,
            dummy_obs,
            str(onnx_path),
            opset_version=17,
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        )
    print(f"ONNX exported → {onnx_path}")

    # ── agent.py ───────────────────────────────────────────────────────────────
    agent_py_path = SUBMISSION_DIR / "agent.py"
    agent_code = '''import os
import numpy as np
import onnxruntime as ort

class Agent:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(current_dir, "model.onnx")
        self.session = ort.InferenceSession(onnx_path)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, obs: dict, info: dict) -> np.ndarray:
        lidar    = (obs["lidar"] / 15.0).astype(np.float32)
        velocity = np.array([obs["velocity"] / 5.0],  dtype=np.float32)
        steering = np.array([obs["steering"] / 0.42], dtype=np.float32)
        return np.expand_dims(np.concatenate([lidar, velocity, steering]), axis=0)

    def postprocess_action(self, raw_action: np.ndarray) -> tuple[float, float]:
        # Denormalize from [-1, 1] policy space → physical bounds (must match env.py).
        a0 = float(np.clip(raw_action[0], -1.0, 1.0))  # target_speed axis
        a1 = float(np.clip(raw_action[1], -1.0, 1.0))  # steering axis
        target_speed = -0.6 + (a0 + 1.0) * (5.0 - (-0.6)) / 2.0   # -> (-0.6 reverse .. 5.0 forward) m/s
        steering     = -0.42 + (a1 + 1.0) * (0.42 - (-0.42)) / 2.0  # -> (-0.42 .. 0.42) rad
        return float(target_speed), float(steering)

    def predict(self, obs: dict, info: dict) -> tuple[float, float]:
        model_input = self.preprocess(obs, info)
        raw = self.session.run([self.output_name], {self.input_name: model_input})[0][0]
        return self.postprocess_action(raw)
'''
    agent_py_path.write_text(agent_code)
    print(f"agent.py written → {agent_py_path}")


if __name__ == "__main__":
    main()