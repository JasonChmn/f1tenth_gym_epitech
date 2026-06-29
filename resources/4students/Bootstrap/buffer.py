"""Replay buffer for DQN training."""

from collections import deque
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples using deque + numpy."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Store a single transition.
        
        Args:
            state: current observation
            action: executed action
            reward: received reward
            next_state: resulting observation
            done: whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Random sample of batch_size transitions.
        
        Uses numpy random choice to pick indices from the deque.
        
        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        # Use np.random.choice on array indices since deque has no .sample()
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)