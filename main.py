import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.drone2d import Drone2D


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


if __name__ == '__main__':
    path = "data/models/torquev3.zip"
    if not os.path.exists(path):
        train(path)
    test(path)
