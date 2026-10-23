import os
import sys

if 'SUMO_HOME' in os.environ:
    sumo_bin = os.path.join(os.environ['SUMO_HOME'], 'bin')
    if os.name == 'nt': 
        os.add_dll_directory(sumo_bin)
else:
    sys.exit("Error: Please declare environment variable 'SUMO_HOME'")

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

if __name__ == "__main__":
    base_env = SumoEnvironment(
        net_file='intersection.net.xml',
        route_file='intersection.rou.xml',
        use_gui=False,
        num_seconds=3600,
        delta_time=10,      
        yellow_time=5,
        min_green=5,
        single_agent=True, 
        add_system_info=False
    )
    env = SumoFix(base_env)

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0005,
        n_steps=512,          
        batch_size=64,        
        ent_coef=0.01        
    )

    model.learn(total_timesteps=200000)
    model.save("traffic_ai_model")
    print("saved")