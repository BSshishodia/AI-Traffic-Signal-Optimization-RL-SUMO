import traci

class TrafficEnv:

    def __init__(self):
        self.sumo_cmd = ["sumo", "-c", "sumo_env/config.sumocfg"]

    def start(self):
        traci.start(self.sumo_cmd)

    def step(self):

        traci.simulationStep()

        # 🔥 USE FIRST JUNCTION (WORKING ONE)
        tls = traci.trafficlight.getIDList()[0] #tls = "J12"

        lanes = list(set(traci.trafficlight.getControlledLanes(tls)))

        queue = 0
        for lane in lanes:
            queue += traci.lane.getLastStepHaltingNumber(lane)

        queue = min(queue, 10)

        state = [queue]
        reward = -queue

        return state, reward

    def close(self):
        traci.close()