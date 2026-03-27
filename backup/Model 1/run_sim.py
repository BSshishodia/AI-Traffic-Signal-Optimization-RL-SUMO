import os
import sys
import subprocess

# 1. Setup SUMO paths
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

def setup_simulation():
    print("--- Preparing Traffic ---")
    # Path to the randomTrips tool inside your SUMO folder
    random_trips = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    
    # Generate random routes for 1000 seconds
    subprocess.call([
        "python", random_trips, 
        "-n", "simulation/test.net.xml", 
        "-r", "simulation/test.rou.xml", 
        "-e", "1000", 
        "--allow-fringe"
    ])
    
    # Create the config file automatically
    with open("simulation/test.sumocfg", "w") as f:
        f.write("""<configuration>
    <input>
        <net-file value="test.net.xml"/>
        <route-files value="test.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1000"/>
    </time>
</configuration>""")
    print("--- Files Ready ---")

def start_sim():
    # Launch SUMO with the GUI
    traci.start(["sumo-gui", "-c", "simulation/test.sumocfg"])
    
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        if step % 100 == 0:
            print(f"Simulation Step: {step}")
            
    traci.close()
    print("Simulation Finished!")

if __name__ == "__main__":
    setup_simulation()
    start_sim()