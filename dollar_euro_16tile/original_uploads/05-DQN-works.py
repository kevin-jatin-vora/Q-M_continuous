import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from ContinuousDollarEuroEnv_radial import ContinuousDollarEuroEnv, ScalarizeReward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# Hyperparameters
# ==============================
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 5e-4  # Target network update rate
LR = 1e-3  # Learning rate
UPDATE_EVERY = 4  # How often to update the network

N_STEP = 200_000
TEST_STEPS = 10_000
TEST_RUNS = 5

STEP_EPS_DECAY = 0.01 ** (1.0 / 200000.0)

RUNS = 5
SAVE_FILE = "dqn_no_pruning.npy"

env = ContinuousDollarEuroEnv(render_mode=None, auto_render=False)

# ==============================
# Neural Network Architecture
# ==============================
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 actions (Down, Up, Left, Right)
        )

    def forward(self, x):
        return self.net(x)

# ==============================
# Replay Buffer
# ==============================
class ReplayBuffer:
    def __init__(self):
        self.mem = deque(maxlen=BUFFER_SIZE)

    def add(self, s, a, r, ns, d):
        self.mem.append((s, a, r, ns, d))

    def sample(self):
        batch = random.sample(self.mem, BATCH_SIZE)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32).to(device),
            torch.tensor(a, dtype=torch.long).unsqueeze(1).to(device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(ns, dtype=torch.float32).to(device),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.mem)

# ==============================
# Agent (Traditional DQN)
# ==============================
class Agent:
    def __init__(self):
        self.q = QNet().to(device)  # Local Q-network
        self.qt = QNet().to(device)  # Target Q-network

        self.opt = optim.Adam(self.q.parameters(), lr=LR)
        self.mem = ReplayBuffer()
        self.t = 0

    # Action selection (epsilon-greedy)
    def act(self, s, eps):
        s = torch.tensor(s).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = self.q(s)

        if random.random() > eps:
            return int(q_vals.argmax(dim=1))  # Choose best action
        return random.choice([0, 1, 2, 3])  # Random action with epsilon probability

    # Step function to store experiences and learn
    def step(self, s, a, r, ns, d):
        self.mem.add(s, a, r, ns, d)
        self.t = (self.t + 1) % UPDATE_EVERY

        if self.t == 0 and len(self.mem) > BATCH_SIZE:
            experiences = self.mem.sample()
            self.learn(experiences)

    # Learn function to update the Q-network
    def learn(self, experiences, gamma=GAMMA):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- #
        # Get Q-targets (using Double DQN)
        # ---------------------------- #
        with torch.no_grad():
            q_local_next = self.q(next_states)     # [B, A]
            q_target_next = self.qt(next_states)   # [B, A]

            best_next_actions = q_local_next.argmax(dim=1, keepdim=True)   # [B, 1]
            q_targets_next = q_target_next.gather(1, best_next_actions)    # [B, 1]
            q_targets = rewards + gamma * q_targets_next * (1 - dones)     # [B, 1]

        # ---------------------------- #
        # Compute the loss
        # ---------------------------- #
        q_all = self.q(states)  # [B, A]
        q_expected = q_all.gather(1, actions)  # [B, 1]

        td_loss = F.mse_loss(q_expected, q_targets)

        self.opt.zero_grad()
        td_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)  # Clip gradients to prevent explosion
        self.opt.step()

        # ---------------------------- #
        # Soft update the target network
        # ---------------------------- #
        self.soft_update(self.q, self.qt, TAU)

    # Soft update function for the target network
    @torch.no_grad()
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.lerp_(local_param.data, tau)

# ==============================
# Environment Setup
# ==============================
def make_env(render=False):
    render_mode = "human" if render else None
    auto = bool(render)
    base_env = ContinuousDollarEuroEnv(render_mode=render_mode, auto_render=auto)
    env = ScalarizeReward(base_env, weights=(1.0, 1.0))
    return env

# ==============================
# Training Loop (Fixed Steps)
# ==============================
def dqn_fixed_steps(agent, env, n_steps=200_000, max_t=200, eps_start=1.0, eps_end=0.05, eps_decay=0.995, test_interval=2_000, test_runs=5):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    state = env.reset()[0]
    score = 0.0
    step_in_ep = 0

    for step in range(1, n_steps + 1):
        step_in_ep += 1
        action = agent.act(state, eps=eps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done or step_in_ep % max_t == 0:
            scores_window.append(score)
            state = env.reset()[0]
            score = 0.0
            step_in_ep = 0
            eps = max(eps_end, eps_decay * eps)

        if step % test_interval == 0:
            avg_test_score = test_agent(agent, test_runs=test_runs, max_t=max_t)
            scores.append(avg_test_score)
            print(f"Step {step:>7} | Avg(100) {np.mean(scores_window):6.3f} | eps {eps:5.3f} | Test {avg_test_score:6.3f}")

    return scores

# ==============================
# Test Agent
# ==============================
def test_agent(agent, test_runs=5, max_t=200, render=False):
    env_test = make_env(render=render)
    test_scores = []
    for _ in range(test_runs):
        state = env_test.reset(seed=0)[0]
        score = 0.0
        done = False
        steps = 0
        while not done and steps < max_t:
            steps += 1
            action = agent.act(state, eps=0.0)  # Greedy policy for testing
            next_state, reward, terminated, truncated, _ = env_test.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
        test_scores.append(score)
    env_test.close()
    return float(np.mean(test_scores))

# ==============================
# Main Function
# ==============================
if __name__ == "__main__":
    env = make_env(render=False)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    runs = 5
    N_STEPS = 150_000
    TEST_EVERY = 2000
    ts = np.zeros((runs, int(N_STEPS / TEST_EVERY)))

    for i in range(runs):
        agent = Agent()
        test_scores = dqn_fixed_steps(
            agent, env,
            n_steps=N_STEPS,
            max_t=100,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=0.995,
            test_interval=TEST_EVERY,
            test_runs=5,
        )

        ts[i] = test_scores
        plt.figure()
        plt.plot(np.arange(len(test_scores)) * TEST_EVERY, test_scores)
        plt.xlabel("Env steps")
        plt.ylabel("Test return")
        plt.show()
        torch.save(agent.q.state_dict(), "dqn.pth")
        np.save("dqn.npy", np.array(ts))