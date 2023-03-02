import imageio
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import envs
from config import Config


def render_drone(model_path, config, simulation_length=1000):
    save_path = model_path.parent.parent / "gifs"

    model = PPO.load(model_path)
    
    env = gym.make("Drone2D", **{ **config["env_kwargs"], **{ "render_mode": "rgb_array", } })
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
        save_path / f"rendered_{model_path.stem}.gif",
        [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
        fps=env.metadata["render_fps"],
    )

    return rewards


def main():
    from pathlib import Path

    save_path = Path("logs/killing_it_multi_2023_03_02_17_34_15")
    config = Config(save_path=save_path)

    print("Rendering")
    model_paths = [folder for folder in (save_path / "models").iterdir()]
    for model_path in model_paths:
        print(model_path)
        rewards = render_drone(model_path=model_path, simulation_length=1000, config=config)


if __name__ == "__main__":
    main()