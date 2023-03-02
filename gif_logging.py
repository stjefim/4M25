from typing import Any, Dict

import imageio
import cv2

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class GifRecorderCallback(BaseCallback):
    def __init__(self, eval_env, save_path, render_freq: int, n_eval_episodes: int=1, deterministic: bool=True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self.save_path = save_path
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic


    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            print(f"Rendering at {self.n_calls}")
            images = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                img = self._eval_env.render()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                images.append(cv2.resize(img, (512, 512)))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            print("Evaluated")
            print(len(images))
            print(images[0].shape)
            print(type(images[0]))
            imageio.mimsave(
                self.save_path / "gifs" / f"rendered_drone_{self.n_calls}_steps.gif",
                [img for i, img in enumerate(images) if i % 2 == 0],
                fps=self._eval_env.metadata["render_fps"],
            )
            print(f"Finished rendering at {self.n_calls}")
        
        return True