from env import TrafficEnv
from rl_agent import RLAgent
import traci

import pandas as pd

env = TrafficEnv()
agent = RLAgent()

EPISODES = 100   #keep 50–100 for good learning

total_rewards = []

for episode in range(EPISODES):

    print(f"\n🚀 Episode {episode+1}")

    env.start()

    state, _ = env.step()

    episode_reward = 0

    for step in range(500):

        action = agent.choose_action(state)

        tls = traci.trafficlight.getIDList()[0]

        current_phase = traci.trafficlight.getPhase(tls)

        if action == 1:
            traci.trafficlight.setPhase(tls, (current_phase + 1) % 4)

        next_state, reward = env.step()

        agent.update(state, action, reward, next_state)

        state = next_state
        episode_reward += reward

    total_rewards.append(episode_reward)

    print(f"Episode Reward: {episode_reward}")
    print(f"Epsilon: {agent.epsilon:.3f}")

    # 🔥 DECAY EXPLORATION
    agent.decay_epsilon()

    env.close()

# =========================
# SAVE RL MODEL
# =========================
agent.save()
print("\n✅ RL model saved")

# =========================
# SHOW RESULTS
# =========================
print("\n📊 Rewards per Episode:")
for i, r in enumerate(total_rewards):
    print(f"Episode {i+1}: {r}")

# after training ends
df = pd.DataFrame({"reward": total_rewards})
df.to_csv("analysis/rl_rewards.csv", index=False)