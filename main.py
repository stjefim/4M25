from datetime import datetime
import logging
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import envs
from trainer import trainer
from evaluator import render_drone
from config import Config


def main():
    keyword = "hovering_with_pole_relative_target"
    save_path = Path("logs") / f"{keyword}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    save_path.mkdir()
    (save_path / "gifs").mkdir()
    config = Config(save_path=save_path)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="",
        handlers=[
            logging.FileHandler(save_path / "logs.log"),
            logging.StreamHandler()
        ]
    )

    model = trainer(save_path=save_path, config=config)

    # Render video
    logging.info("Logging started")
    now = datetime.now()
    model_paths = [folder for folder in (save_path / "models").iterdir()]
    for model_path in model_paths:
        print(model_path)
        rewards = render_drone(model_path=model_path, simulation_length=1000, config=config)
    logging.info("Logging finished")
    logging.info(f"Logging duration: {datetime.now() - now}")

    # Evaluating train model
    # TODO: tweak the parameters here
    logging.info("Evaluating model")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    logging.info(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    logging.info("Evaluation finished")


if __name__ == '__main__':
    main()
