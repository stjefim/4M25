from pathlib import Path
from datetime import datetime


def Config(save_path):
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
        "total_timesteps": 20_000,
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

    env_kwargs = {
        "render_mode": None,
        "action_type": 0, # ACTION_FORCES in drone2d
        "multiple_obj": True,
        "reward_func": lambda *args: -100*((args[0][0] - 2.5 - args[1][0]) ** 2 + (args[0][1] - 0.25 - args[1][1]) ** 2), # args[0] == drone position, args[1] == target position
        
        # ------------------------------ REWARD FUNCTIONS ------------------------------
        # reward = -np.exp((self.drone.position[0] - 2.5 - self.target[0]) ** 2 + (self.drone.position[1] - 0.25 - self.target[1]) ** 2) # exponential does not work
        # reward = -100*((self.drone.position[0] - 2.5 - self.target[0]) ** 2 + (self.drone.position[1] - 0.25 - self.target[1]) ** 2)
        # reward = 1. / (0.1 + np.sqrt((self.drone.position[0] - 2.5 - self.target[0]) ** 2 + ( self.drone.position[1] - 0.25 - self.target[1]) ** 2)) # Best?
        # -----------------------------------------------------------------------------
    }
    
    return {
        "n_envs": n_envs,
        "policy_args": policy_args,
        "training_args": training_args,
        "gif_recording_args": gif_recording_args,
        "checkpointing_args": checkpointing_args,
        "env_kwargs": env_kwargs,
    }

keyword = "baseline"
save_path = Path("logs") / f"{keyword}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
save_path.mkdir()
config = Config(save_path=save_path)