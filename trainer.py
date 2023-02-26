from pathlib import Path
from datetime import datetime
import logging
import json

# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import envs.drone2d


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
        "total_timesteps": 10_000_000,
        "progress_bar": True,
    }

    return n_envs, policy_args, training_args


def train(save_path, policy_args, training_args, n_envs=8):
    # Create a vectorized environment with the "Drone2D" environment
    env = make_vec_env("Drone2D", n_envs=n_envs)

    # Create a PPO model with the provided policy arguments and the vectorized environment
    model = PPO("MlpPolicy", env, **policy_args)
    model.learn(**training_args) # train

    model.save(save_path / "model.zip")
    logging.info(f"Model saved to {save_path / 'model.zip'}")

    return model


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

    # Hyperparamaters
    n_envs, policy_args, training_args = get_parameters()
    logging.info(f"{n_envs=}")
    logging.info(f"policy_args={json.dumps(policy_args, indent=4)}")
    logging.info(f"training_args={json.dumps(training_args, indent=4)}")

    # Training
    logging.info("Starting training")
    now = datetime.now()
    model = train(
        n_envs=n_envs, policy_args=policy_args,
        training_args=training_args, save_path=save_path,
    )
    logging.info("Training finished")
    logging.info(f"Training duration: {datetime.now() - now}")

    # Evaluating train model
    # TODO: tweak the parameters here 
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    logging.info(f"{mean_reward=}, {std_reward=}")


if __name__ == "__main__":
    main()