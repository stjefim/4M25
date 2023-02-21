from gymnasium.envs.registration import register

register(
     id="Drone2D",
     entry_point="envs.drone2d:Drone2D",
     max_episode_steps=1000
)
