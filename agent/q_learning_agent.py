import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Deep Reinforcement Learning Agent for Autonomous Pricing
    """
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def store(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.from_numpy(np.array(states)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            q_next = self.model(next_states).max(1)[0]

        target = rewards + self.gamma * q_next
        loss = self.loss_fn(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
