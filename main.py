from datetime import datetime
import logging
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from envs.drone2d import Drone2D
from trainer import trainer
from evaluator import render_drone


def test(path):
    model = PPO.load(path)
    env = gym.make("Drone2D", render_mode="human", action_type=Drone2D.ACTION_FORCE_AND_TORQUE)
    obs, info = env.reset(seed=0)

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def main():
    # Save path
    keyword = "baseline"
    save_path = Path("logs") / f"{keyword}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    save_path.mkdir()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="",
        handlers=[
            logging.FileHandler(save_path / "logs.log"),
            logging.StreamHandler()
        ]
    )

    trainer(save_path=save_path)

    render_drone()


if __name__ == '__main__':
    main()