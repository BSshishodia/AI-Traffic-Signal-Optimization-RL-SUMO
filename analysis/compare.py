import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("analysis/baseline.csv")
rl = pd.read_csv("analysis/rl_output.csv")

# QUEUE COMPARISON
plt.figure()
plt.plot(baseline["queue"], label="Without RL")
plt.plot(rl["queue"], label="With RL")
plt.title("Queue Comparison")
plt.xlabel("Time Step")
plt.ylabel("Queue Length")
plt.legend()
plt.show()

# VEHICLE COMPARISON
plt.figure()
plt.plot(baseline["vehicles"], label="Without RL")
plt.plot(rl["vehicles"], label="With RL")
plt.title("Vehicle Flow Comparison")
plt.xlabel("Time Step")
plt.ylabel("Vehicles")
plt.legend()
plt.show()