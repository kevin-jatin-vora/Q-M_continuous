import numpy as np
import torch


class ReplayBuffer:
    """Contiguous numpy ring buffer. Sampling copies a batch; adding is O(1)."""

    def __init__(self, size: int = 100_000, state_dim: int = 2):
        self.capacity = int(size)
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._index = 0
        self._size = 0

    def add(self, s, a, r, ns, done):
        i = self._index
        self.states[i] = s
        self.next_states[i] = ns
        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.dones[i] = float(done)
        self._index = (i + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size: int, device):
        idx = np.random.choice(self._size, size=int(batch_size), replace=False)
        return (
            torch.tensor(self.states[idx], dtype=torch.float32, device=device),
            torch.tensor(self.actions[idx], dtype=torch.long, device=device).unsqueeze(1),
            torch.tensor(self.rewards[idx], dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(self.next_states[idx], dtype=torch.float32, device=device),
            torch.tensor(self.dones[idx], dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return self._size
