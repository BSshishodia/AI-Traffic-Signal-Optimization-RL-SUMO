import os
import sys
import pandas as pd
import subprocess

# 1. Setup SUMO Paths
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("CRITICAL: Please declare environment variable 'SUMO_HOME'")

import traci

def generate_route_file():
    """Generates random traffic. The --allow-fringe flag is the 'fix' for small maps."""
    print("\n--- Phase 1: Generating Random Traffic Routes ---")
    random_trips_path = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    
    # Command with fixes for small networks
    cmd = [
        "python", random_trips_path,
        "-n", "simulation/test.net.xml",
        "-r", "simulation/test.rou.xml",
        "-e", "3600",   # 1 hour of traffic
        "-p", "1.5",    # A car every 1.5 seconds
        "--allow-fringe", # <--- Allows cars to start at the map edges
        "--fringe-factor", "10",
        "--validate"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("✅ Routes Generated Successfully: simulation/test.rou.xml")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating routes: {e}")
        sys.exit(1)

def create_sumo_config():
    """Creates the configuration file that links the map and the cars."""
    config_content = """<configuration>
    <input>
        <net-file value="test.net.xml"/>
        <route-files value="test.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <report>
        <no-warnings value="true"/>
    </report>
</configuration>"""
    with open("simulation/test.sumocfg", "w") as f:
        f.write(config_content)
    print("✅ Config Created: simulation/test.sumocfg")

def run_data_collection():
    # Setup files
    generate_route_file()
    create_sumo_config()

    # Start SUMO-GUI so you can verify the cars are moving
    print("\n--- Phase 2: Starting Simulation & Data Collection ---")
    traci.start(["sumo-gui", "-c", "simulation/test.sumocfg"])
    
    data = []
    lane_ids = traci.lane.getIDList()
    
    # Run for 3600 seconds
    for step in range(3600):
        traci.simulationStep()
        
        # Record data every 60 steps (1 minute)
        if step % 60 == 0:
            for lane in lane_ids:
                if ":" not in lane: # Ignore internal intersection lanes
                    v_count = traci.lane.getLastStepVehicleNumber(lane)
                    v_speed = traci.lane.getLastStepMeanSpeed(lane)
                    
                    data.append({
                        "step": step,
                        "lane_id": lane,
                        "vehicle_count": v_count,
                        "avg_speed": round(v_speed, 2)
                    })
        
        if step % 500 == 0:
            print(f"Simulation Progress: {step}/3600 steps...")

    traci.close()
    
    # Save the dataset
    if not os.path.exists('data'):
        os.makedirs('data')
        
    df = pd.DataFrame(data)
    df.to_csv("data/traffic_data.csv", index=False)
    print(f"\n✅ PROJECT UPDATE: 'traffic_data.csv' saved with {len(df)} rows.")
    print("You now have the dataset needed to train the Deep Learning 'Oracle'.")

if __name__ == "__main__":
    run_data_collection()