import pytest
import wrappers
import gym
import gym_minigrid


class WrapperTest:
    @pytest.fixture(scope='function')
    def env(self):
        env = gym.make('MiniGrid-FourRooms-v0')
        return env


class TestSymbolicWrapper(WrapperTest):
    @pytest.fixture(scope='function')
    def wrapper(self, env):
        wrapper = wrappers.SymbolicWrapper(env)
        return wrapper

    def test_observation_space_reset(self, wrapper):
        obs_space = wrapper.observation_space
        obs = wrapper.reset()
        assert obs_space.contains(obs)

    def test_observation_space_step(self, wrapper):
        obs_space = wrapper.observation_space
        wrapper.reset()
        obs = wrapper.step(1)[0]
        assert obs_space.contains(obs)

    def test_observation_space_width(self, wrapper, env):
        obs_space = wrapper.observation_space
        width = env.grid.width
        assert width == obs_space.nvec[0]

    def test_observation_space_height(self, wrapper, env):
        obs_space = wrapper.observation_space
        height = env.grid.height
        assert height == obs_space.nvec[1]


class TestSimpleActionsWrapper(WrapperTest):
    @pytest.fixture(scope='function')
    def wrapper(self, env):
        return wrappers.SimpleActionsWrapper(env)

    @staticmethod
    def step_from(action, x_start, y_start, wrapper):
        wrapper.unwrapped.agent_pos = x_start, y_start
        wrapper.step(action)[0]
        return wrapper.agent_pos

    def test_up_wall(self, wrapper):
        wrapper.reset()
        x, y = 1, 1
        x_new, y_new = self.step_from(0, x, y, wrapper)
        assert x == x_new
        assert y == y_new

    def test_up(self, wrapper):
        wrapper.reset()
        x, y = 1, 2
        x_new, y_new = self.step_from(0, x, y, wrapper)
        assert x == x_new
        assert (y - 1) == y_new

    def test_down_wall(self, wrapper):
        wrapper.reset()
        x, y = 1, wrapper.height - 2
        x_new, y_new = self.step_from(2, x, y, wrapper)
        assert x == x_new
        assert y == y_new

    def test_down(self, wrapper):
        wrapper.reset()
        x, y = 1, 2
        x_new, y_new = self.step_from(2, x, y, wrapper)
        assert x == x_new
        assert (y + 1) == y_new

    def test_left_wall(self, wrapper):
        wrapper.reset()
        x, y = 1, 1
        x_new, y_new = self.step_from(3, x, y, wrapper)
        assert x == x_new
        assert y == y_new

    def test_left(self, wrapper):
        wrapper.reset()
        x, y = 2, 1
        x_new, y_new = self.step_from(3, x, y, wrapper)
        assert (x - 1) == x_new
        assert y == y_new

    def test_right_wall(self, wrapper):
        wrapper.reset()
        x, y = wrapper.width - 2, 1
        x_new, y_new = self.step_from(1, x, y, wrapper)
        assert x == x_new
        assert y == y_new

    def test_right(self, wrapper):
        wrapper.reset()
        x, y = 1, 1
        x_new, y_new = self.step_from(1, x, y, wrapper)
        assert (x + 1) == x_new
        assert y == y_new
