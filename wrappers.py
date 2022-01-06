import gym
from gym.spaces import MultiDiscrete
import gym_minigrid

__all__ = ['SymbolicWrapper', 'StepwiseRewardWrapper', 'SimpleActionsWrapper']


class SymbolicWrapper(gym.Wrapper):
    def __init__(self, env: gym_minigrid.envs.MiniGridEnv, directions=True):
        super().__init__(env)
        self.env = env
        self.directions = directions
        self.observation_space = MultiDiscrete([
            self.env.width, self.env.height, 4]) if self.directions else MultiDiscrete(
            [self.env.width, self.env.height])

    def _symb_state(self):
        if self.directions:
            return [*self.env.agent_pos, self.env.agent_dir]
        return [*self.env.agent_pos]

    def step(self, action):
        # return self.env.step(action)
        _, reward, done, info = self.env.step(action)
        return self._symb_state(), reward, done, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._symb_state()


class StepwiseRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym_minigrid.envs.MiniGridEnv, muted=False):
        super().__init__(env)
        self.env = env
        self.muted = muted

    def step(self, action):
        # return self.env.step(action)
        obs, reward, done, info = self.env.step(action)
        return obs, -(1 - (reward > 0)) * (1 - self.muted), done, info



class SimpleActionsWrapper(gym.Wrapper):
    def __init__(self, env: gym_minigrid.envs.MiniGridEnv):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(4)

    def step(self, action):
        self._apply_direction(action)
        return self.env.step(2)

    def _apply_direction(self, action):
        ACTION_TO_DIRECTION = {
            0: 3,  # ^
            1: 0,  # >
            2: 1,  # V
            3: 2  # <
        }

        if not (0 <= action <= 3):
            raise ValueError('Invalid action chosen. Action must be integer in range [0, 3].')

        self.unwrapped.agent_dir = ACTION_TO_DIRECTION[action]
