import numpy as np
import matplotlib.pyplot as plt

revenue = np.load("revenue_history.npy")
window = 20
moving_avg = np.convolve(
    revenue, np.ones(window) / window, mode="valid"
)

plt.figure(figsize=(10, 5))
plt.plot(revenue, alpha=0.3, label="Episode Revenue")
plt.plot(moving_avg, linewidth=2, label="Moving Average (20 episodes)")
plt.xlabel("Episodes")
plt.ylabel("Total Revenue")
plt.title("Training Revenue Over Episodes")
plt.legend()
plt.grid(True)
plt.show()
