import abc
from abc import ABC
from typing import Optional, Tuple
import torch
import numpy as np

from stable_baselines3.common.policies import BasePolicy


class ModelBasedPolicy(BasePolicy, ABC):
    def _predict(self, observation: torch.Tensor, deterministic: bool = False,
                 belief: Optional[np.ndarray] = None,
                 last_state: Optional = None, last_action: Optional = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        last_action_ = self.last_action if last_action is None else last_action
        last_state_ = self.last_state if last_state is None else last_state
        belief_ = self.belief if belief is None else belief

        belief_, state_new = self._infer_state(observation, last_action_, belief_, last_state_)
        action = self._action_for(belief_, state_new, deterministic)

        if last_state is None:
            self.last_state = state_new
        if last_action is None:
            self.last_action = action
        if belief is None:
            self.belief = belief_
        if belief is not None and last_state is not None:
            return action, belief_, state_new

        return self.output_action(action)

    def output_action(self, action):
        return action

    def reset_state(self):
        self.belief = torch.zeros(1, self.world_model.belief_size, device=self.world_model.device)
        self.last_state = torch.zeros(1, self.world_model.state_size, device=self.world_model.device)
        self.last_action = torch.zeros(1, self.world_model.action_size, device=self.world_model.device)

    def _infer_state(self, observation, action, belief, state_prev):
        return self.world_model.infer_state(observation, action, belief, state_prev)

    @abc.abstractmethod
    def _action_for(self, belief, state, deterministic):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
