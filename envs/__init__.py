from gymnasium.envs.registration import register

register(
     id="Drone2D",
     entry_point="envs.drone2d:Drone2D",
     max_episode_steps=1000
)
register(
     id="DronePole2D",
     entry_point="envs.drone_pole2d:DronePole2D",
     max_episode_steps=1000
)
register(
     id="Drone2DRobust",
     entry_point="envs.drone2d_robust:Drone2DRobust",
     max_episode_steps=1000
)

register(
     id="DronePole2DRobust",
     entry_point="envs.drone_pole2d_robust:DronePole2DRobust",
     max_episode_steps=1000
)
