import imageio
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from envs.drone2d import Drone2D
from config import config


def render_drone(save_path, config, simulation_length=1000):
    model = PPO.load(save_path / "models" / "final_model.zip")
    
    env = gym.make("Drone2D", render_mode="rgb_array", action_type=Drone2D.ACTION_FORCE_AND_TORQUE, reward_func=config["env_kwargs"]["reward_func"], multiple_obj=config["env_kwargs"]["multiple_obj"], initial_target_pos=config["env_kwargs"]["initial_target_pos"])
    obs, info = env.reset(seed=0)

    images = [env.render()]
    rewards = []
    for _ in range(simulation_length+1):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        images.append(env.render())

        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

    imageio.mimsave(
        save_path / "rendered_drone.gif",
        [np.array(img) for i, img in enumerate(images) if i%2 == 0],
        fps=env.metadata["render_fps"],
    )

    return rewards


def main():
    from pathlib import Path

    save_path = Path("logs/test_2023_03_02_16_50_09")
    
    rewards = render_drone(save_path=save_path, simulation_length=1000, config=config)
    print(rewards)



if __name__ == "__main__":
    main()