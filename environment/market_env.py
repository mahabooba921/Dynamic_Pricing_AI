import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DynamicPricingEnv(gym.Env):
    """
    Autonomous Market Environment for Dynamic Pricing
    """

    def __init__(self):
        super(DynamicPricingEnv, self).__init__()

        # State: [current_price, demand, inventory]
        self.observation_space = spaces.Box(
            low=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 1000.0, 1000.0], dtype=np.float32),
            dtype=np.float32
        )

        # Actions: decrease price, keep price, increase price
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.price = np.random.uniform(10, 50)
        self.demand = np.random.uniform(100, 300)
        self.inventory = 500

        return self._get_obs(), {}

    def step(self, action):
        # Adjust price
        if action == 0:
            self.price *= 0.9
        elif action == 2:
            self.price *= 1.1

        # Demand reacts to price
        self.demand = max(0, 500 - self.price * np.random.uniform(5, 8))

        sales = min(self.demand, self.inventory)
        revenue = sales * self.price
        self.inventory -= sales

        reward = revenue

        done = self.inventory <= 0

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.array([self.price, self.demand, self.inventory], dtype=np.float32)
