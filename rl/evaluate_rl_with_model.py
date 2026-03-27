import traci
import pandas as pd
from rl_agent import RLAgent

agent = RLAgent()
agent.load()

sumo_cmd = ["sumo", "-c", "sumo_env/config.sumocfg"]
traci.start(sumo_cmd)

total_queue = []
total_time = []

state = [0]

for step in range(500):

    action = agent.choose_action(state)

    tls = traci.trafficlight.getIDList()[0]
    current_phase = traci.trafficlight.getPhase(tls)

    if action == 1:
        traci.trafficlight.setPhase(tls, (current_phase + 1) % 4)

    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()
    total_time.append(len(vehicles))

    queue = 0
    lanes = list(set(traci.trafficlight.getControlledLanes(tls)))
    for lane in lanes:
        queue += traci.lane.getLastStepHaltingNumber(lane)

    queue = min(queue, 10)
    state = [queue]

    total_queue.append(queue)

traci.close()

pd.DataFrame({
    "queue": total_queue,
    "vehicles": total_time
}).to_csv("analysis/rl_output.csv", index=False)

print("✅ RL output saved")