import imageio
import numpy as np
import random
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import envs
from config import Config
from reward_funcs import _check_terminate

FORCING_INTERVAL = 150
FORCE_MAGNITUDE = 10

def render_drone(model_path, config, simulation_length=1000):
    save_path = model_path.parent.parent / "gifs"
    save_path = Path("figures") / "gifs"

    model = PPO.load(model_path)

    env = gym.make(config["env_name"], **{ **config["env_kwargs"], **{ "render_mode": "rgb_array", } })
    obs, info = env.reset(seed=0)

    images = [env.render()]
    rewards = []
    for i in range(1, simulation_length+1):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if it time to apply forcing
        if i % FORCING_INTERVAL == 0:
            print("Siii")
            force = np.random.normal(size=2)
            force = force / np.linalg.norm(force) * FORCE_MAGNITUDE
            env.drone.ApplyForceToCenter(force=force, wake=True)

        
        rewards.append(reward)
        images.append(env.render())

        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

    imageio.mimsave(
        save_path / f"stable_rendered_{model_path.stem}_{FORCE_MAGNITUDE}_int_{FORCING_INTERVAL}.gif",
        [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
        duration=simulation_length / env.metadata["render_fps"],
    )

    return rewards


def main():
    from pathlib import Path

    save_path = Path("figures")
    config = Config(save_path=save_path)
    model_path = Path("logs/stability/final_model.zip")
    rewards = render_drone(model_path=model_path, simulation_length=1000, config=config)

    return

    print("Rendering")
    model_paths = [folder for folder in (save_path / "models").iterdir()]
    for model_path in model_paths:
        print(model_path)
        rewards = render_drone(model_path=model_path, simulation_length=1000, config=config)


if __name__ == "__main__":
    main()