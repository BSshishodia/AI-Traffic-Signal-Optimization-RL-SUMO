import traci
import pandas as pd

sumo_cmd = ["sumo", "-c", "sumo_env/config.sumocfg"]

traci.start(sumo_cmd)

total_queue = []
total_time = []

for step in range(500):

    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()
    total_time.append(len(vehicles))

    queue = 0
    for tls in traci.trafficlight.getIDList():
        lanes = list(set(traci.trafficlight.getControlledLanes(tls)))
        for lane in lanes:
            queue += traci.lane.getLastStepHaltingNumber(lane)

    total_queue.append(queue)

traci.close()

pd.DataFrame({
    "queue": total_queue,
    "vehicles": total_time
}).to_csv("analysis/baseline.csv", index=False)

print("✅ Baseline saved")