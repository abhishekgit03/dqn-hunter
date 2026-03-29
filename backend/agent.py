"""
Deep Q-Network (DQN) Agent
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque

# Hyperparameters
STATE_DIM    = 23       
ACTION_DIM   = 16       

BUFFER_SIZE  = 50_000    # how many transitions to store
BATCH_SIZE   = 64        # how many to sample per learning step
GAMMA        = 0.95     # discount factor — how much future reward matters
                         #     0.99 = thinks ~100 steps ahead; try 0.95 for faster but greedier
LR           = 1e-4     # learning rate — lower = more stable but slower
TAU          = 0.005     # soft target network update rate (explained below)

EPSILON_START = 1.0      # start fully random
EPSILON_END   = 0.10   
EPSILON_DECAY = 0.997   # multiply epsilon by this each episode
                        

LEARN_EVERY  = 4         # only train every N steps (more stable than every step)
TARGET_UPDATE = 100      # hard-update target net every N steps (alternative to TAU)

CHECKPOINT_DIR = "checkpoints"

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)   # auto-drops oldest when full

    def push(self, state, action, reward, next_state, done):
        """Store one transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        """Sample a random batch. Returns tensors ready for the network."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# DQN

class DQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)   


# DQN Agent

class DQNAgent:
    def __init__(self, device=None):
        self.device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Two identical networks
        self.online_net = DQN().to(self.device)   # trained every step
        self.target_net = DQN().to(self.device)   
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()                     

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.buffer    = ReplayBuffer()

        self.epsilon   = EPSILON_START
        self.steps     = 0    

    def act(self, state, training=True):
        """
        Epsilon-greedy: random action with prob epsilon, else greedy.
        During evaluation (training=False) always greedy.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return q_values.argmax(dim=1).item()

    def get_q_values(self, state):
        """Return all 17 Q-values for a state (used by heatmap endpoint)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_t).squeeze(0).cpu().numpy()

    # Memory

    def remember(self, state, action, reward, next_state, done):
        """Push one transition into the replay buffer."""
        self.buffer.push(state, action, reward, next_state, float(done))
        self.steps += 1

        # Learn every LEARN_EVERY steps, but only once buffer is big enough
        if self.steps % LEARN_EVERY == 0 and len(self.buffer) >= BATCH_SIZE:
            self._learn()

        # Soft-update target network every step
        self._soft_update_target()

    def _learn(self):
        """
        One gradient update step.

        The loss is:  MSE( Q(s,a),  r + γ * max_a' Q_target(s', a') )

                           ↑ predicted       ↑ Bellman target

        We only update Q for the action that was actually taken.
        The other 16 Q-values aren't touched this step.
        """
        if len(self.buffer) < BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample()
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Current Q-values for actions that were taken
        # .gather(1, actions.unsqueeze(1)) picks Q(s, a) for the specific action taken
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: r + γ * max Q_target(s', a')  [0 if terminal]
        with torch.no_grad():
            next_q      = self.target_net(next_states).max(1)[0]
            target_q    = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients (rare here but good practice)
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def _soft_update_target(self):
        """
        Blend target network weights toward online network weights.
        θ_target = TAU * θ_online + (1 - TAU) * θ_target
        TAU=0.005 means target moves 0.5% toward online each step.
        Smoother than hard-copying every N steps.
        """
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                TAU * online_param.data + (1 - TAU) * target_param.data
            )

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    # Checkpoints

    def save(self, episode, path=None):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = path or os.path.join(CHECKPOINT_DIR, f"dqn_ep{episode}.pt")
        torch.save({
            "episode":          episode,
            "online_state":     self.online_net.state_dict(),
            "target_state":     self.target_net.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "epsilon":          self.epsilon,
            "steps":            self.steps,
        }, path)
        print(f"  ✓ Saved checkpoint → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        print(f"  ✓ Loaded checkpoint ← {path}  (ep {ckpt['episode']}, ε={self.epsilon:.3f})")
        return ckpt["episode"]

    def load_latest(self):
        """Auto-load the most recent checkpoint if one exists."""
        if not os.path.exists(CHECKPOINT_DIR):
            return 0
        files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
        if not files:
            return 0
        latest = max(files, key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)))
        return self.load(os.path.join(CHECKPOINT_DIR, latest))

