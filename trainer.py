from pathlib import Path
from datetime import datetime
import logging
import json

# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

import envs
from config import Config


def train_model(config, save_path):
    env = make_vec_env("Drone2D", n_envs=config["n_envs"], env_kwargs=config["env_kwargs"])

    model = PPO("MlpPolicy", env, tensorboard_log=save_path, **config["policy_args"])

    # Save a checkpoint every 1000 steps
    callback = [CheckpointCallback(**config["checkpointing_args"])]
    model.learn(callback=callback, **config["training_args"])

    model.save(save_path / "models" / "final_model.zip")
    logging.info(f"Model saved to {save_path / 'model.zip'}")
    return model


def trainer(save_path, config):
    # Hyperparamaters
    logging.info(f"n_envs={config['n_envs']}")
    logging.info(f"policy_args={json.dumps(config['policy_args'], indent=4)}")
    logging.info(f"training_args={json.dumps(config['training_args'], indent=4)}")
    logging.info(f"checkpointing_args={config['checkpointing_args']}")
    logging.info(f"env_kwargs={config['env_kwargs']}")

    # Training
    logging.info("Starting training")
    now = datetime.now()
    model = train_model(config=config, save_path=save_path)
    logging.info("Training finished")
    logging.info(f"Training duration: {datetime.now() - now}")

    return model


def main():
    keyword = "killing_it_multi"
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


if __name__ == "__main__":
    main()