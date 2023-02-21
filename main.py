import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import envs.drone2d


def run_env():
    env = gym.make("Drone2D", render_mode="human")
    obs, info = env.reset()

    for _ in range(1000):
        action = [0, 0]
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


def train(path):
    env = make_vec_env("Drone2D", n_envs=4)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save(path)


def test(path):
    model = PPO.load(path)
    env = gym.make("Drone2D", render_mode="human")
    obs, info = env.reset(seed=0)

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


if __name__ == '__main__':
    path = "data/models/v1.zip"
    if not os.path.exists(path):
        train(path)
    test(path)
