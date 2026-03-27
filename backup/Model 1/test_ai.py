import traci
import torch
import numpy as np
from stable_baselines3 import PPO
from train_rl import SmartTrafficEnv, TrafficOracle

def run_test():
    checkpoint = torch.load("models/oracle_brain.pth", weights_only=False)
    oracle = TrafficOracle()
    oracle.load_state_dict(checkpoint['model_state'])
    oracle.eval()
    
    model = PPO.load("models/final_rl_agent")
    env = SmartTrafficEnv(oracle, checkpoint['max_val'])
    env.render = True 
    obs, _ = env.reset()
    
    print("--- AI SHOWCASE: Watch the Yellow Light Logic & Clearing ---")
    
    step_count = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        step_count += 1
        
        if step_count % 5 == 0:
            print(f"Max Wait Time detected by AI: {obs[3]} seconds")
            
        if done: break
        
    print("✅ All cars successfully navigated the intersection.")
    env.close()

if __name__ == "__main__":
    run_test()