from datetime import datetime
import logging
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from envs.drone2d import Drone2D
<<<<<<< HEAD
from trainer import trainer
from evaluator import render_drone
=======


def run_env():
    env = gym.make("Drone2D", render_mode="human", action_type=Drone2D.ACTION_FORCE_AND_TORQUE)
    obs, info = env.reset()

    for _ in range(1000): # number of steps we run the environment for
        # take a zero action
        action = [0, 0]
        obs, reward, terminated, truncated, info = env.step(action)

        # reset environment if episode ended
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


def train(path):
    env = make_vec_env("Drone2D", n_envs=4, env_kwargs={"action_type": Drone2D.ACTION_FORCE_AND_TORQUE})
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(path)
>>>>>>> 636dfadab2c764eb0c79605fcc14fca7712031d8


def test(path):
    model = PPO.load(path)
    
    # create environment and reset it
    env = gym.make("Drone2D", render_mode="human", action_type=Drone2D.ACTION_FORCE_AND_TORQUE)
    obs, info = env.reset(seed=0)

    for _ in range(1000): # number of steps we run the environment for at test time
        # predict an action using the model
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
<<<<<<< HEAD
    main()
=======
    path = "data/models/torquev3.zip"
    if not os.path.exists(path):
        train(path)
    test(path)
>>>>>>> 636dfadab2c764eb0c79605fcc14fca7712031d8
