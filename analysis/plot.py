import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("analysis/rl_rewards.csv")

plt.plot(df["reward"])
plt.title("RL Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()