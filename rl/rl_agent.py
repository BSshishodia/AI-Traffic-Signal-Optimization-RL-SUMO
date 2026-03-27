import random
import numpy as np
import pickle

class RLAgent:

    def __init__(self):
        self.q_table = {}
        self.actions = [0, 1]  # 0 = keep, 1 = switch

        self.alpha = 0.1
        self.gamma = 0.9

        # 🔥 Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        key = self.get_state_key(state)

        if key not in self.q_table:
            self.q_table[key] = [0, 0]

        return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, next_state):

        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if key not in self.q_table:
            self.q_table[key] = [0, 0]

        if next_key not in self.q_table:
            self.q_table[next_key] = [0, 0]

        q_old = self.q_table[key][action]
        q_next = max(self.q_table[next_key])

        self.q_table[key][action] = q_old + self.alpha * (reward + self.gamma * q_next - q_old)

    # 🔥 DECAY EXPLORATION
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename="rl/q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename="rl/q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)