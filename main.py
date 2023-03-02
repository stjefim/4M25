from datetime import datetime
import logging
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from envs.drone2d import Drone2D
from trainer import trainer
from evaluator import render_drone


def main():
    # Save path
    keyword = "cancel_gravity"
    save_path = Path("logs") / f"{keyword}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    save_path.mkdir()
    (save_path / "gifs").mkdir()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="",
        handlers=[
            logging.FileHandler(save_path / "logs.log"),
            logging.StreamHandler()
        ]
    )

    model = trainer(save_path=save_path)

    # Render video
    rewards = render_drone(save_path=save_path, simulation_length=1000)

    # Evaluating train model
    # TODO: tweak the parameters here 
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    logging.info(f"{mean_reward=}, {std_reward=}")


if __name__ == '__main__':
    main()
