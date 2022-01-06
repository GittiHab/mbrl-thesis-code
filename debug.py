from typing import NamedTuple, Union, Tuple

import numpy as np

from algos.PlaNet.planet import PlaNet
import torch
from torch.nn.functional import one_hot

__all__ = ['UninferredState', 'State', 'Transition', 'EnvDebug', 'ManualEval', 'manual_eval']

from algos.PlaNet.policies import MPCPlanner


class UninferredState(NamedTuple):
    belief: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    device: Union[str, torch.device]

    @property
    def previous_state(self):
        return State(self.belief, self.state)


class State(NamedTuple):
    belief: torch.Tensor
    state: torch.Tensor


class Transition(NamedTuple):
    current_state: State
    action: torch.Tensor
    next_observation: tuple
    next_state: State

    @property
    def merged_states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.stack([self.current_state.belief, self.next_state.belief]), \
               torch.stack([self.current_state.state, self.next_state.state])


class EnvDebug:
    def __init__(self, world_model, env):
        self._env = env
        self.belief_size = world_model.belief_size
        self.state_size = world_model.state_size
        self.action_size = world_model.action_size
        self.device = world_model.device
        self.world_model = world_model

    def _initial_state(self):
        return UninferredState(self._empty_tensor(self.belief_size, self.device),
                               self._empty_tensor(self.state_size, self.device),
                               self._empty_tensor(self.action_size, self.device),
                               self.device)

    def _empty_state(self):
        return State(self._empty_tensor(self.belief_size, self.device),
                     self._empty_tensor(self.state_size, self.device))

    def _retrieve(self, x, y, state: UninferredState, as_state=False):
        return self._infer_state(x, y, state.action, state.previous_state, as_state)

    def _infer_state(self, x, y, action, state: State, as_state=False):
        b, s = self.world_model.infer_state(self._observation(x, y, self.device), action, state.belief, state.state)
        if as_state:
            return State(b, s)
        return b, s

    @staticmethod
    def _observation(x, y, device):
        return torch.tensor([[x, y]], dtype=torch.float32, device=device)

    @staticmethod
    def _empty_tensor(dimension, device):
        return torch.zeros(1, dimension, device=device)


class ManualEval(EnvDebug):

    def __init__(self, env, model: PlaNet, config):
        super().__init__(model.policy.world_model, env)
        self.model = model
        self.policy = self.model.policy  # type: MPCPlanner
        self.device = self.model.device

        self.max_timesteps = config.max_timesteps
        self.render = 'render' in config and config.render
        self.episodes = config.episodes

    def reset(self):
        state_init = self._initial_state()
        return state_init.previous_state, state_init.action

    def _one_hot(self, action):
        return one_hot(action, num_classes=self._env.action_space.n)

    def _action_as_tensor(self, action):
        return self._one_hot(torch.tensor([action], device=self.device))

    def convert_action(self, action):
        return action

    def ask_for_action(self):
        action = input('Action: ')
        while action not in list('0123'):
            action = input('Action (0/1/2/3): ')
        return int(action)

    def eval(self):
        obs = self._env.reset()
        done = False
        rewards = []
        episode = 0

        previous_state, previous_action = self.reset()
        for i in range(self.max_timesteps):

            current_state = self._infer_state(obs[0], obs[1], previous_action, previous_state, True)

            transition = Transition(previous_state, previous_action, obs, current_state)
            reward_int = self.policy.reward_space.rewards(*transition.merged_states, transition.action.unsqueeze(0))

            print('Current location:    ', obs)
            print('Reward Space Reward: ', reward_int.item())
            if self.policy.uses_value_model:
                value = self.policy.reward_space.values(current_state.belief, current_state.state)
                print('Reward Space Value:  ', value.item())
            print('====================')

            action = self.convert_action(self.ask_for_action())

            obs, reward, done, info = self._env.step(action)
            print('Environment Reward:  ', reward)

            previous_state = current_state
            previous_action = self._action_as_tensor(action)

            if self.render:
                self._env.render()

            rewards.append(reward)
            if done:
                if self.policy.uses_value_model:
                    current_state = self._infer_state(obs[0], obs[1], previous_action, previous_state, True)
                    value = self.policy.reward_space.values(current_state.belief, current_state.state)
                    print('Reward Space Value:  ', value.item())
                obs = self._env.reset()
                previous_state, previous_action = self.reset()
                if hasattr(self.model.policy, 'reset_state'):
                    self.model.policy.reset_state()
                episode += 1
                if episode == self.episodes:
                    break

        mean_return = np.sum(rewards) / episode
        return mean_return, episode


def manual_eval(env, model: PlaNet, config):
    # .render() doesn't like me and only works at random times (for me random)
    if 'render' in config and config.render:
        env.reset()
        env.render()
    eval = ManualEval(env, model, config)
    return eval.eval()
