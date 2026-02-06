# Dynamic Pricing using Reinforcement Learning
An AI-based dynamic pricing system that learns optimal pricing strategies using reinforcement learning to maximize long-term revenue.

## Project Overview
This project implements an autonomous AI decision-making system for dynamic pricing.  
A reinforcement learning agent interacts with a simulated market environment and learns how to adjust prices based on demand and inventory conditions in order to maximize total revenue over time.
Unlike traditional machine learning models that rely on historical labeled data, this system learns through continuous interaction with the environment using reward-based feedback.

## Why This Project?
Dynamic pricing is widely used in industries such as e-commerce, airlines, ride-hailing, and hotels.This project demonstrates how **Reinforcement Learning (AI)** can be applied to pricing problems where decisions directly impact future outcomes.

## Technologies Used
- Python
- NumPy
- Gymnasium
- PyTorch
- Matplotlib
- Reinforcement Learning (Q-Learning)


## Project Structure
Dynamic_Pricing_AI/
│
├-agent/
│ └── q_learning_agent.py # Q-learning agent logic
│
├── environment/
│ └── market_env.py # Market simulation environment
│
├── train.py # Trains the RL agent
├── visualize.py # Plots revenue vs episodes
├── revenue_history.npy # Saved training revenue data
├── requirements.txt # Dependencies
└── README.md

## System Architecture
The diagram below illustrates the architecture of our **Dynamic Pricing AI system**.  
It shows the flow of the Q-learning agent interacting with the market environment:
![Architecture Diagram](docs/architecture_diagram.png)

**Explanation of components:**
- **Market Environment:** Simulates real-world market data including pricing, demand, and competitors.  
- **State:** Represents the current market situation, inventory, and pricing conditions.  
- **Agent:** The AI model (Q-learning) that decides optimal pricing strategies.  
- **Action:** Adjusts product prices based on agent decisions.  
- **Reward:** Measures performance (e.g., profit, revenue, sales).  
- **Next State:** The environment updates according to the action, creating a continuous feedback loop.


## How the System Works
- The environment simulates market conditions such as demand and inventory.
- The agent observes the current state and selects a pricing action.
- The environment returns a reward based on revenue.
- The agent updates its Q-values using the Q-learning algorithm.
- Over multiple episodes, the agent learns an optimal pricing strategy.


## ▶ How to Run the Project
### 1. Install dependencies
```bash
pip install -r requirements.txt
python train.py
python visualize.py

## Output:
After training, the system generates revenue values for each episode and saves them in a NumPy file.
A visualization is created to show how total revenue improves over training episodes, demonstrating that the agent learns an optimal pricing strategy over time.

## AI vs ML Explanation:
This project is categorized as Artificial Intelligence (AI) because it uses Reinforcement Learning, where an agent learns by interacting with an environment and receiving rewards.
No labeled dataset is used, and decisions are optimized through trial and error rather than prediction.

## Interview Summary:
This project demonstrates autonomous decision-making, reinforcement learning implementation, and real-world applicability in dynamic pricing and demand control systems.
