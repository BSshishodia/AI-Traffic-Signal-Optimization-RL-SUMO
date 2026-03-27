import gymnasium as gym
import numpy as np
import traci
import time

class TrafficEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render = render
        self.action_space = gym.spaces.Discrete(2) # 0 = NS Green, 1 = EW Green
        # Observation: [NS_Queue, EW_Queue, Current_Phase, Max_Wait_Time]
        self.observation_space = gym.spaces.Box(low=0, high=2000, shape=(4,), dtype=np.float32)
        self.tls_id = None 
        self.current_phase = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sumo_cmd = ["sumo-gui" if self.render else "sumo", "-c", "simulation/test.sumocfg", "--start", "--no-warnings"]
        try: traci.close()
        except: pass
        traci.start(sumo_cmd)
        
        all_lights = traci.trafficlight.getIDList()
        self.tls_id = all_lights[0] if all_lights else "bs_0"
        self.current_phase = 0
        return np.array([0, 0, 0, 0], dtype=np.float32), {}

    def step(self, action):
        # --- 1. REALISTIC PHYSICS: The Yellow Light Transition ---
        if action != self.current_phase:
            # If changing phases, we MUST show a yellow light first
            yellow_phase = "yyyrrryyyrrr" if self.current_phase == 0 else "rrryyyrrryyy"
            traci.trafficlight.setRedYellowGreenState(self.tls_id, yellow_phase)
            for _ in range(4): # 4 seconds of yellow
                traci.simulationStep()
                if self.render: time.sleep(0.01)

        # --- 2. THE GREEN LIGHT ---
        green_phase = "GGgrrrGGgrrr" if action == 0 else "rrrGGgrrrGGg"
        traci.trafficlight.setRedYellowGreenState(self.tls_id, green_phase)
        self.current_phase = action
        
        for _ in range(12): # 12 seconds of green
            traci.simulationStep()
            if self.render: time.sleep(0.01)
            
        # --- 3. GATHER METRICS ---
        lanes = [l for l in traci.lane.getIDList() if ":" not in l]
        ns_lanes = [l for l in lanes if 'N' in l or 'S' in l]
        ew_lanes = [l for l in lanes if 'E' in l or 'W' in l]
        
        ns_q = sum([traci.lane.getLastStepVehicleNumber(l) for l in ns_lanes])
        ew_q = sum([traci.lane.getLastStepVehicleNumber(l) for l in ew_lanes])
        
        # --- 4. STARVATION KILLER: Find the single angriest driver ---
        max_wait = 0
        for lane in lanes:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
            for v in veh_ids:
                w = traci.vehicle.getWaitingTime(v)
                if w > max_wait:
                    max_wait = w
                    
        total_wait = sum([traci.lane.getWaitingTime(l) for l in lanes])
        
        # RESEARCH REWARD: Punish overall wait, but heavily punish the MAX wait
        # If one car sits for 100 seconds, the penalty is huge (-500)
        reward = -(total_wait * 0.1 + max_wait * 5.0)
        
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        # New State shape
        obs = np.array([float(ns_q), float(ew_q), float(self.current_phase), float(max_wait)], dtype=np.float32)
        return obs, reward, done, False, {}

    def close(self):
        try: traci.close()
        except: pass