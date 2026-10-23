import gymnasium as gym

from sumo_rl import SumoEnvironment

from stable_baselines3 import PPO

import numpy as np




class SumoFix(gym.Wrapper):

  def reset(self, **kwargs):

    result = self.env.reset(**kwargs)

    obs, info = result if isinstance(result, tuple) else (result, {})

    return np.array(obs, dtype=np.float32).flatten(), info

  def step(self, action):

    obs, reward, terminated, truncated, info = self.env.step(action)

    return np.array(obs, dtype=np.float32).flatten(), float(reward), terminated, truncated, info



import time

def run_demo():
    base_env = SumoEnvironment(
        net_file='intersection.net.xml',
        route_file='intersection.rou.xml',
        use_gui=True,        
        num_seconds=3600,
        delta_time=10,
        yellow_time=5,
        min_green=5,
        single_agent=True
    )
    env = SumoFix(base_env)

    model = PPO.load("traffic_ai_model")

    obs, info = env.reset()
    
    print("Demo started. Adjust Delay to 100ms, then press 'Play' in SUMO-GUI!")
    
    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        print(f"Step {step} | AI chose phase: {action}") 
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        time.sleep(0.1) 
        
        if terminated or truncated:
            print("Simulation ended.")
            break
    ts = base_env.traffic_signals[list(base_env.traffic_signals.keys())[0]]
    print("lanes:", ts.lanes)                  # -> INCOMING_LANES
    print("green phases:", ts.green_phases)    # -> GREEN_PHASE_LANES
    print("min_green/yellow/delta:", ts.min_green, ts.yellow_time, ts.delta_time)

    import sumolib
    net = sumolib.net.readNet("intersection.net.xml")
    for lane_id in ts.lanes:
        print(lane_id, net.getLane(lane_id).getLength())   # -> LANE_LENGTH_METERS
    env.close()



if __name__ == "__main__":

  run_demo()