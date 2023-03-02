from pathlib import Path
from datetime import datetime
import logging
import json

# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

import envs.drone2d
<<<<<<< HEAD

=======
>>>>>>> 165b0407dfc694c1275c6b0a812920c6b9c64339
from gif_logging import GifRecorderCallback
from config import config


def train_model(config, save_path):
    env = make_vec_env("Drone2D", n_envs=config["n_envs"], env_kwargs=config["env_kwargs"])

    model = PPO("MlpPolicy", env, tensorboard_log=save_path, **config["policy_args"])

    # Save a checkpoint every 1000 steps
    callback = [CheckpointCallback(**config["checkpointing_args"])]
    if config["gif_recording_args"]["save_gif"]:
<<<<<<< HEAD
        callback.append(
            GifRecorderCallback(env, save_path=save_path, render_freq=config["gif_recording_args"]["save_freq"])
        )
=======
        callback = GifRecorderCallback(env, render_freq=config["gif_recording_args"]["save_freq"])
>>>>>>> 165b0407dfc694c1275c6b0a812920c6b9c64339
    model.learn(callback=callback, **config["training_args"])

    model.save(save_path / "models" / "final_model.zip")
    logging.info(f"Model saved to {save_path / 'model.zip'}")
    return model


def trainer(save_path):
    # Hyperparamaters
    logging.info(f"n_envs={config['n_envs']}")
    logging.info(f"policy_args={json.dumps(config['policy_args'], indent=4)}")
    logging.info(f"training_args={json.dumps(config['training_args'], indent=4)}")

    # Training
    logging.info("Starting training")
    now = datetime.now()
    model = train_model(config=config, save_path=save_path)
    logging.info("Training finished")
    logging.info(f"Training duration: {datetime.now() - now}")

    return model


def main():
    # Save path
    keyword = "baseline"
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


if __name__ == "__main__":
    main()