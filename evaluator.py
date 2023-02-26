
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder

from envs.drone2d import Drone2D


def render_drone(save_path, simulation_length=1000):
    model = PPO.load(save_path / "model.zip")
    
    env = gym.make("Drone2D", render_mode="human", action_type=Drone2D.ACTION_FORCE_AND_TORQUE)
    # env = VecVideoRecorder(
    #     env, save_path, record_video_trigger=lambda x: x == 0,
    #     video_length=simulation_length,
    # )
    obs, info = env.reset(seed=0)

    rewards = []
    for _ in range(simulation_length+1):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)

        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

    return rewards


def main():
    from pathlib import Path

    save_path = Path("logs/baseline_2023_02_26_14_46_22")
    
    rewards = render_drone(save_path=save_path, simulation_length=200)
    print(rewards)



if __name__ == "__main__":
    main()