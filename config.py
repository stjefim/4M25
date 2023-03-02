from pathlib import Path
from datetime import datetime

import reward_funcs


def Config(save_path):
    n_envs = 8
    env_name = "DronePole2D"

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
        "total_timesteps": 1_000_000,
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
        "action_type": 1,  # ACTION_FORCES in drone2d
        "multiple_obj": False,
        "reward_func": reward_funcs.PlusOnePole(),
        "initial_target_pos": [-1, 3]
    }
    
    return {
        "env_name": "DronePole2D",
        "n_envs": n_envs,
        "policy_args": policy_args,
        "training_args": training_args,
        "gif_recording_args": gif_recording_args,
        "checkpointing_args": checkpointing_args,
        "env_kwargs": env_kwargs,
    }


keyword = "test"
save_path = Path("logs") / f"{keyword}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
save_path.mkdir()
(save_path / "gifs").mkdir()
config = Config(save_path=save_path)
