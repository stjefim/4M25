from pathlib import Path
from datetime import datetime
import logging
import json

# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

import envs.drone2d
from gif_logging import GifRecorderCallback


def get_parameters(save_path):
    n_envs = 8

    # TODO: maybe move this to argparse instead
    # TODO: add the other parameters
    policy_args = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "verbose": 1,
    }

    training_args = {
        "total_timesteps": 100_000,
        "progress_bar": True,
    }

    gif_recording_args = {
        "save_gif": False,
        "save_freq": training_args["total_timesteps"] // 10,
    }

    checkpointing_args = {
        "save_freq": max((training_args["total_timesteps"] // 10) // n_envs, 1),
        "save_path": save_path / "models",
        "name_prefix": "model",
        "save_replay_buffer": True,
        "save_vecnormalize": False, # adjust if VecNormalize is used
    }

    return {
        "n_envs": n_envs,
        "policy_args": policy_args,
        "training_args": training_args,
        "gif_recording_args": gif_recording_args,
        "checkpointing_args": checkpointing_args,
    }


def train_model(args, save_path):
    env = make_vec_env("Drone2D", n_envs=args["n_envs"])

    model = PPO("MlpPolicy", env, tensorboard_log=save_path, **args["policy_args"])

    # Save checkpoints
    callback = [CheckpointCallback(**args["checkpointing_args"])]
    if args["gif_recording_args"]["save_gif"]:
        callback.append(
            GifRecorderCallback(env, save_path=save_path, render_freq=args["gif_recording_args"]["save_freq"])
        )
    model.learn(callback=callback, **args["training_args"])

    model.save(save_path / "models" / "final_model.zip")
    logging.info(f"Model saved to {save_path / 'model.zip'}")

    return model


def trainer(save_path):
    # Hyperparamaters
    args = get_parameters(save_path=save_path)
    logging.info(f"n_envs={args['n_envs']}")
    logging.info(f"policy_args={json.dumps(args['policy_args'], indent=4)}")
    logging.info(f"training_args={json.dumps(args['training_args'], indent=4)}")

    # Training
    logging.info("Starting training")
    now = datetime.now()
    model = train_model(args=args, save_path=save_path)
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