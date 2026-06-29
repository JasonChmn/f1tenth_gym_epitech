"""DQN agent with epsilon-greedy selection and target network."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import QNetwork


class DQNAgent:
    """Deep Q-Network agent.

    - Uses epsilon-greedy action selection
    - Hard update of target network
    - MSE loss on Bellman target = r + gamma * max(Q_target(s'))
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
    ):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Init DQN")
        self.device = torch.device("cpu")
        self.action_dim = action_dim

        # online network
        self.policy_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # target network (hard update copies weights from policy_net)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps_step = 0  # counts how many update steps (episodes) have run

    def select_action(self, state: np.array, episode: int = 0):
        """Epsilon-greedy action selection.

        Args:
            state: current observation (numpy array)
            episode: current episode number (controls epsilon decay)

        Returns:
            action (int)
        """
        # decay epsilon over episodes
        eps = self.eps_end + (self.eps_start - self.eps_end) * (self.eps_decay ** episode)

        if np.random.random() < eps:
            return np.random.randint(self.action_dim)  # explore

        # exploit: pick action with highest Q-value
        with torch.no_grad():
            q_values = self.policy_net(
                torch.tensor(state, dtype=torch.float32).to(self.device)
            )
        return q_values.argmax(dim=-1).cpu().item()

    def update(self, states, actions, rewards, next_states, dones):
        """Compute and perform one training step.

        Bellman target: r + gamma * max(Q_target(next_state))
        (zero target if episode ended)
        """
        # convert to tensors
        s = torch.tensor(states).float().to(self.device)
        a = torch.tensor(actions).long().to(self.device)
        r = torch.tensor(rewards).float().to(self.device)
        ns = torch.tensor(next_states).float().to(self.device)
        dn = torch.tensor(dones).float().to(self.device)

        # current Q values for taken actions
        q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # target: r + gamma * max(Q_target(next_state)) (0 if done)
        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1)[0]
            target = r + self.gamma * next_q * (1 - dn)

        # MSE loss and step
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        """Hard update: copy policy_net weights to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())