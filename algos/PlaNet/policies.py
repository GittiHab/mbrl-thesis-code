from abc import ABC
from math import inf
from typing import Optional, Tuple

import numpy as np
import torch
from gym import Env
from gym.spaces import Discrete
from stable_baselines3.common.policies import BasePolicy
from torch.distributions.categorical import Categorical
from torch.nn.functional import one_hot

from algos.PlaNet.world_model import DreamerModel
from algos.PlaNet.reward_space import RewardSpace, setup_reward_space

__all__ = ['DiscreteMPCPlanner', 'ContinuousMPCPlanner', 'MPCPlanner']

from algos.commons.policies import ModelBasedPolicy


class MPCPlanner(ModelBasedPolicy):
    '''
    Model-predictive control planner with cross-entropy method and learned transition model.
    Originally from PlaNet project by Kaixhin published under the MIT license, this code has been modified.
    https://github.com/Kaixhin/PlaNet/blob/dacf418bd2760675e32ab90b1f1a149dee3eeaac/planner.py
    '''

    @staticmethod
    def from_args(config, env: Env, world_model: DreamerModel, lr_step_size=100):
        reward_space = setup_reward_space(config, world_model, lr_step_size, 'value_real')
        args = (config.planning_horizon, config.optimisation_iters, config.candidates, config.top_candidates,
                world_model, env.observation_space, env.action_space, reward_space)
        kwargs = {'weighted': config.weighted}
        if isinstance(env.action_space, Discrete):
            return DiscreteMPCPlanner(*args, **kwargs)
        else:
            return ContinuousMPCPlanner(*args, **kwargs)

    def __init__(self, planning_horizon, optimisation_iters, candidates, top_candidates, world_model: DreamerModel,
                 observation_space, action_space, reward_space: RewardSpace,
                 min_action=-inf, max_action=inf, squash_output: bool = False, weighted: bool = False, **kwargs):
        super().__init__(observation_space, action_space, squash_output, **kwargs)
        action_size = action_space.n if isinstance(action_space, Discrete) else action_space.shape[0]
        self.world_model = world_model
        self.reward_space = reward_space

        self.weighted = weighted

        self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates
        self.reset_state()

    def set_reward_space(self, reward_space: RewardSpace):
        self.reward_space = reward_space

    @property
    def device(self) -> torch.device:
        return self.world_model.device

    def _action_for(self, belief, state, deterministic):
        return self.sample(belief, state)

    @property
    def uses_value_model(self):
        return self.reward_space.has_value_model

    def sample(self, belief, state):
        batch_size, belief_size, state_size = belief.size(0), belief.size(1), state.size(1)
        belief = belief.unsqueeze(dim=1).expand(batch_size, self.candidates, belief_size).reshape(-1, belief_size)
        state = state.unsqueeze(dim=1).expand(batch_size, self.candidates, state_size).reshape(-1, state_size)

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        theta = self.init_theta(batch_size, belief.device)
        for _ in range(self.optimisation_iters):
            actions = self.sample_actions(batch_size, *theta)
            # Sample next states
            beliefs, states = self.world_model.transition_model(
                state, self.transform_actions(actions), belief, return_start=True)
            # Calculate expected returns (technically sum of rewards over planning horizon)
            if self.uses_value_model:
                returns = self.reward_space.rewards(
                    beliefs,
                    states,
                    self.transform_actions(actions))
                # last_values = self.values(beliefs[-1], states[-1]).view(-1)
                last_values = self.reward_space.values(beliefs[-1], states[-1]).view(-1)

                if self.world_model.pcont:
                    pcont = self.world_model.pcont_model(
                        beliefs.view(-1, belief_size),
                        states.view(-1, state_size)).view(self.planning_horizon, -1).cumsum(dim=0)
                    returns *= pcont[:-1]
                    last_values *= pcont[-1]
                returns = returns.sum(dim=0)
                returns += last_values
            else:
                returns = self.reward_space.rewards(
                    beliefs,
                    states,
                    self.transform_actions(actions)).sum(dim=0)
            # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(batch_size, self.candidates).topk(
                self.top_candidates, dim=1, largest=True, sorted=False)
            # Fix indices for unrolled actions
            topk += self.candidates * torch.arange(0, batch_size, dtype=torch.int64, device=topk.device).unsqueeze(1)
            theta = self.update_distribution(actions, topk, batch_size, returns, *theta)

        # Return first action mean µ_t
        return self.final_action(*theta)

    def final_action(self, *theta):
        raise NotImplementedError()

    def transform_actions(self, actions):
        raise NotImplementedError()

    def init_theta(self, batch_size, device):
        raise NotImplementedError()

    def sample_actions(self, batch_size, *theta):
        raise NotImplementedError()

    def update_distribution(self, actions, topk, batch_size, returns, *theta):
        raise NotImplementedError()


class ContinuousMPCPlanner(MPCPlanner):

    def final_action(self, action_mean, action_std_dev):
        return action_mean[0].squeeze(dim=1)

    def transform_actions(self, actions):
        return actions

    def init_theta(self, batch_size, device):
        return [torch.zeros(self.planning_horizon, batch_size, 1, self.action_size, device=device),
                torch.ones(self.planning_horizon, batch_size, 1, self.action_size, device=device)]

    def sample_actions(self, batch_size, action_mean, action_std_dev):
        # Sample actions (time x (batch x candidates) x actions):
        actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, batch_size, self.candidates,
                                                              self.action_size, device=action_mean.device)) \
            .view(self.planning_horizon, batch_size * self.candidates, self.action_size)
        actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
        return actions

    def update_distribution(self, actions, topk, batch_size, returns, action_mean, action_std_dev):
        _, topk = returns.reshape(batch_size, self.candidates) \
            .topk(self.top_candidates, dim=1, largest=True, sorted=False)
        # Fix indices for unrolled actions:
        topk += self.candidates * torch.arange(0, batch_size, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)
        best_actions = actions[:, topk.view(-1)] \
            .reshape(self.planning_horizon, batch_size, self.top_candidates, self.action_size)
        # Update belief with new means and standard deviations
        return [best_actions.mean(dim=2, keepdim=True),
                best_actions.std(dim=2, unbiased=False, keepdim=True)]


class DiscreteMPCPlanner(MPCPlanner):
    eps = 0.001  # ensure non-zero probabilities

    def output_action(self, action):
        return action.argmax(-1)

    def final_action(self, probabilities):
        return one_hot(probabilities[0].squeeze(dim=1).argmax(-1), self.action_size).type(torch.float)

    def transform_actions(self, actions):
        return one_hot(actions, self.action_size).type(torch.float)

    def init_theta(self, batch_size, device):
        return [torch.ones(self.planning_horizon, batch_size, 1, self.action_size, device=device, dtype=torch.float)]

    def sample_actions(self, batch_size, probabilities):
        dist = Categorical(
            probabilities.expand(self.planning_horizon, batch_size, self.candidates, self.action_size))
        # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
        return dist.sample().view(self.planning_horizon, batch_size * self.candidates)

    def update_distribution(self, actions, topk, batch_size, returns, probabilities):
        if self.weighted:
            return self._weighted_update(actions, topk, batch_size, returns, probabilities)
        return self._normal_update(actions, topk, batch_size, returns, probabilities)

    def _weighted_update(self, actions, topk, batch_size, returns, probabilities):
        best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, batch_size, self.top_candidates)
        best_returns = returns[topk.view(-1)] \
            .reshape(1, batch_size, self.top_candidates) \
            .expand(self.planning_horizon, batch_size, self.top_candidates)
        best_returns = best_returns - best_returns.min() + self.eps  # ensure that it's non-zero and positive
        # Update belief with new means and standard deviations
        total_returns = best_returns.sum(2)  # sum over candidates
        for i in range(self.action_size):
            probabilities[:, :, :, i] = ((best_returns * (best_actions == i)).sum(2) / total_returns).unsqueeze(-1)
        return [probabilities]

    def _normal_update(self, actions, topk, batch_size, returns, probabilities):
        best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, batch_size, self.top_candidates)
        for i in range(self.action_size):
            probabilities[:, :, :, i] = ((best_actions == i).sum(2) / topk.size(-1)).unsqueeze(-1)
        return [probabilities]
