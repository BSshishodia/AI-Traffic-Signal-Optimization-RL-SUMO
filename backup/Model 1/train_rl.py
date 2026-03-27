import os
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from traffic_env import TrafficEnv

class TrafficOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, 64, 2, batch_first=True)
        self.fc = torch.nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_oracle():
    checkpoint = torch.load("models/oracle_brain.pth", weights_only=False)
    model = TrafficOracle()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['max_val']

class SmartTrafficEnv(TrafficEnv):
    def __init__(self, oracle, max_val):
        super().__init__()
        self.oracle = oracle
        self.max_val = max_val
        self.history = []
        # UPGRADE: 5 inputs -> [ns_q, ew_q, current_phase, max_wait, future_prediction]
        self.observation_space = gym.spaces.Box(low=0, high=2000, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.history = []
        obs, info = super().reset(seed=seed)
        return np.append(obs, 0.0).astype(np.float32), info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        # Total cars for the Oracle
        total_q = obs[0] + obs[1]
        self.history.append(total_q)
        
        pred = 0.0
        if len(self.history) >= 10:
            seq = torch.FloatTensor(self.history[-10:]).view(1, 10, 1) / self.max_val
            with torch.no_grad(): 
                pred = self.oracle(seq).item() * self.max_val
                
        # Combine base environment observation with Oracle prediction
        final_obs = np.array([obs[0], obs[1], obs[2], obs[3], float(pred)], dtype=np.float32)
        return final_obs, reward, done, truncated, info

if __name__ == "__main__":
    oracle_model, max_v = load_oracle()
    env = SmartTrafficEnv(oracle_model, max_v)
    
    # RESEARCH SETTINGS: 25k is the sweet spot for this new logic
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=0.0003,
                n_steps=1024,
                batch_size=64,
                ent_coef=0.05) # Balanced exploration
    
    print("--- Training Ultimate AI Agent (25,000 steps) ---")
    model.learn(total_timesteps=25000)
    model.save("models/final_rl_agent")
    env.close()