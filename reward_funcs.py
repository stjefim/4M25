import numpy as np


def _calc_distance(observation, target):
    """Calculate distance from the drone to the target

    observation: tuple(x, y, angle, ...)
    """
    return np.linalg.norm(np.array(observation)[:2] - np.array(target))


def _check_terminate(observation):
    return observation[0] > 2.5 or observation[0] < -2.5 or observation[1] > 4.75 or observation[2] > np.pi / 2\
             or observation[2] < -np.pi / 2 or observation[3] > np.pi / 4 or observation[3] < -np.pi / 4 or observation[1] <= 0.1


class MinusDistance:
    def __init__(self, scale=10, terminate_reward=-1000, offset=7):
        self.scale = scale
        self.terminate_reward = terminate_reward
        self.offset = offset

    def __call__(self, observation, target, *args, **kwargs):
        if _check_terminate(observation):
            return self.terminate_reward, True
        return self.scale * (self.offset - _calc_distance(observation, target)), False


class InverseDistance:
    def __init__(self, epsilon=0.1, terminate_reward=-100):
        self.epsilon = epsilon
        self.terminate_reward = terminate_reward

    def __call__(self, observation, target, *args, **kwargs):
        if _check_terminate(observation):
            return self.terminate_reward, True
        return 1. / (_calc_distance(observation, target) + self.epsilon), False


class ExpDistance:
    def __init__(self, terminate_reward=-100):
        self.terminate_reward = terminate_reward

    def __call__(self, observation, target, *args, **kwargs):
        if _check_terminate(observation):
            return self.terminate_reward, True
        return -np.exp(_calc_distance(observation, target)), False


class PlusOnePole:
    def __init__(self, terminate_reward=-100):
        self.terminate_reward = terminate_reward

    def __call__(self, observation, target, *args, **kwargs):
        if _check_terminate(observation):
            return self.terminate_reward, True
        return 1, False



if __name__ == '__main__':
    observation = [2, 2, 0, -1, -1, -1, -1, -1]
    target = [3, 3]
    reward = MinusDistance()
    print(reward(observation, target))

    observation = [2, 2, np.pi / 2 + 0.1, -1, -1, -1, -1, -1]
    print(reward(observation, target))
