import unittest

import pytest
import gym
import gym_minigrid
from tests.utils.true_model import *
import torch
from torch.nn.functional import one_hot
import wrappers


def assert_tensors_equal(expected, given):
    assert expected.numpy().tolist() == given.numpy().tolist()


class ModelTest:
    @pytest.fixture(scope='function')
    def env(self):
        env = gym.make('MiniGrid-Empty-6x6-v0')
        env = wrappers.SymbolicWrapper(wrappers.SimpleActionsWrapper(env))
        return env

    @staticmethod
    def apply_actions(env, actions):
        return [env.step(a) for a in actions][-1]


class TestTransitionModel(ModelTest):
    @pytest.fixture()
    def model(self, env):
        return TrueTransitionModel(env.spec.id)

    @staticmethod
    def predict_assert(model, start_state, action, target_state):
        state_pred = model(torch.tensor(start_state), one_hot(torch.tensor(action), num_classes=3))
        assert target_state == state_pred[1].tolist()

    def step_assert(self, model, env, start_state, action):
        s1, r1, t1, _ = env.step(action)

        self.predict_assert(model, start_state, action, s1)

    def test_two_actions(self, model, env):
        s0 = env.reset()
        a0 = 2
        s1, r0, t0, _ = env.step(a0)
        a1 = 1
        s2, r2, t2, _ = env.step(a1)

        s1_pred = model(torch.tensor(s0), one_hot(torch.tensor(a0), num_classes=3))
        s2_pred = model(torch.tensor(s1), one_hot(torch.tensor(a1), num_classes=3))

        assert s1 == s1_pred[1].tolist()
        assert s2 == s2_pred[1].tolist()

    def test_step_right(self, model, env):
        s0 = env.reset()
        self.step_assert(model, env, s0, 1)

    def test_step_left(self, model, env):
        env.reset()
        s0, r0, t0, _ = self.apply_actions(env, [1, 3])

        self.step_assert(model, env, s0, 2)

    def test_step_down(self, model, env):
        env.reset()
        s0, r0, t0, _ = self.apply_actions(env, [2])

        self.step_assert(model, env, s0, 2)

    def test_step_up(self, model, env):
        env.reset()
        s0, r0, t0, _ = self.apply_actions(env, [2, 0])

        self.step_assert(model, env, s0, 2)

    def test_run_into_wall(self, model, env):
        env.reset()
        s0, r0, t0, _ = self.apply_actions(env, [0])

        self.step_assert(model, env, s0, 2)


class TestRewardModel(ModelTest):
    @pytest.fixture()
    def model(self, env):
        return TrueRewardModel(env.spec.id)

    @staticmethod
    def predict_assert(model, start_state, has_reward):
        reward_pred = model(torch.tensor(start_state)).item()
        if has_reward:
            assert reward_pred != 0
        else:
            assert reward_pred == 0

    def test_no_reward_initial(self, env, model):
        s0 = env.reset()

        self.predict_assert(model, s0, False)

    def test_no_reward_step(self, env, model):
        env.reset()
        s1, r1, t1, _ = self.apply_actions(env, [1, 2])
        self.predict_assert(model, s1, False)

    def test_reward(self, env, model):
        env.reset()
        s_final, r_f, t_f, _ = self.apply_actions(env, [2, 2, 2, 1, 1, 1])

        self.predict_assert(model, s_final, True)


class TestTerminalModel(ModelTest):
    @pytest.fixture()
    def model(self, env):
        return TrueTerminalModel(env.spec.id)

    @staticmethod
    def predict_assert(model, start_state, is_terminal):
        terminal_pred = model(torch.tensor(start_state))
        if is_terminal:
            assert terminal_pred
        else:
            assert not terminal_pred

    def test_non_terminal_initial(self, env, model):
        s0 = env.reset()

        self.predict_assert(model, s0, torch.tensor([True]))

    def test_non_terminal_step(self, env, model):
        env.reset()
        s1, r1, t1, _ = self.apply_actions(env, [1, 2])
        self.predict_assert(model, s1, torch.tensor([True]))

    def test_terminal(self, env, model):
        env.reset()
        s_final, r_f, t_f, _ = self.apply_actions(env, [2, 2, 2, 1, 1, 1])

        self.predict_assert(model, s_final, torch.tensor([False]))


if __name__ == '__main__':
    unittest.main()
