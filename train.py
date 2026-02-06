from environment.market_env import DynamicPricingEnv
from agent.q_learning_agent import DQNAgent
import numpy as np

def train():
    env = DynamicPricingEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    episodes = 300
    revenue_history = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.store(state, action, reward, next_state)
            agent.train()

            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            print(f"Episode {episode} | Total Revenue: {int(total_reward)}")

        revenue_history.append(total_reward)

    np.save("revenue_history.npy", revenue_history)
    print("Training history saved!")

if __name__ == "__main__":
    train()

