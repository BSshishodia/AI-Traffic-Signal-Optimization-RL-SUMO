import traci
import csv


sumo_cmd = ["sumo-gui", "-c", "sumo_env/config.sumocfg"]
#sumo_cmd = ["sumo", "-c", "sumo_env/config.sumocfg"] -> for dataset generation
traci.start(sumo_cmd)


file = open("traffic_dataset.csv", "w", newline="")
writer = csv.writer(file)

writer.writerow(["step", "tls_id", "queue"])


MAX_STEPS = 2000

step = 0

while step < MAX_STEPS:

    traci.simulationStep()

    if step < 400:
        traci.simulation.setScale(0.5)   # LOW
    elif step < 800:
        traci.simulation.setScale(1.0)   # MEDIUM
    elif step < 1200:
        traci.simulation.setScale(1.5)   # HIGH
    elif step < 1600:
        traci.simulation.setScale(1.0)   # MEDIUM
    else:
        traci.simulation.setScale(0.6)   # LOW


    tls_ids = traci.trafficlight.getIDList()

    for tls in tls_ids:
        try:
            lanes = list(set(traci.trafficlight.getControlledLanes(tls)))

            queue = 0
            for lane in lanes:
                queue += traci.lane.getLastStepHaltingNumber(lane)

            writer.writerow([step, tls, queue])

        except Exception as e:
            continue

    if step % 100 == 0:
        print(f"Step {step} running...")

    step += 1


traci.close()
file.close()

print("✅ Large dataset created: traffic_dataset.csv")