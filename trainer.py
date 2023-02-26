
from pathlib import Path
from datetime import datetime
import logging
import json

# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import envs.drone2d
from video_logging import VideoRecorderCallback


def get_parameters():
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

    video_recording_args = {
        "save_video": False,
        "save_frequency": training_args["total_timesteps"] // 10,
    }

    return {
        "n_envs": n_envs,
        "policy_args": policy_args,
        "training_args": training_args,
        "video_recording_args": video_recording_args,
    }


def train_model(args, save_path):
    env = make_vec_env("Drone2D", n_envs=args["n_envs"])

    model = PPO("MlpPolicy", env, tensorboard_log=save_path, **args["policy_args"])

    callback = None
    if args["video_recording_args"]["save_video"]:
        callback = VideoRecorderCallback(env, render_freq=args["video_recording_args"]["save_frequency"])
    model.learn(callback=callback, **args["training_args"])

    model.save(save_path / "model.zip")
    logging.info(f"Model saved to {save_path / 'model.zip'}")

    return model


def trainer(save_path):
    # Hyperparamaters
    args = get_parameters()
    logging.info(f"n_envs={args['n_envs']}")
    logging.info(f"policy_args={json.dumps(args['policy_args'], indent=4)}")
    logging.info(f"training_args={json.dumps(args['training_args'], indent=4)}")

    # Training
    logging.info("Starting training")
    now = datetime.now()
    model = train_model(args=args, save_path=save_path)
    logging.info("Training finished")
    logging.info(f"Training duration: {datetime.now() - now}")


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


if __name__ == "__main__":
    main()