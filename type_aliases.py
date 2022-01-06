from typing import NamedTuple, Optional
import torch
from stable_baselines3.common.type_aliases import ReplayBufferSamples

__all__ = ['StatePredictions', 'ImaginedTrajectories', 'Cell', 'ExtendedReplayBufferSamples']


class ExtendedReplayBufferSamples(ReplayBufferSamples):
    _all_observations: Optional[torch.Tensor] = None

    @staticmethod
    def from_replay_data(samples):
        return ExtendedReplayBufferSamples(*samples)

    @property
    # TODO: this does not always work. Luckily, it seems that the MiniGrid environment works such that this should work out.
    #       Still, right now this is required due to legacy code.
    #       However, I would like to point to the to do that I marked "ID1".
    def all_observations(self) -> torch.Tensor:
        if self._all_observations is None:
            self._all_observations = torch.cat([self.observations[0].unsqueeze(0), self.next_observations])
        return self._all_observations


class StatePredictions(NamedTuple):
    beliefs: torch.Tensor
    prior_states: torch.Tensor
    prior_means: torch.Tensor
    prior_std_devs: torch.Tensor
    posterior_states: torch.Tensor
    posterior_means: torch.Tensor
    posterior_std_devs: torch.Tensor

    @property
    def states(self):
        return self.posterior_states

    def prepend(self, other):
        return StatePredictions(beliefs=torch.cat([other.beliefs, self.beliefs]),
                                prior_states=torch.cat([other.prior_states, self.prior_states]),
                                prior_means=torch.cat([other.prior_means, self.prior_means]),
                                prior_std_devs=torch.cat([other.prior_std_devs, self.prior_std_devs]),
                                posterior_states=torch.cat([other.posterior_states, self.posterior_states]),
                                posterior_means=torch.cat([other.posterior_means, self.posterior_means]),
                                posterior_std_devs=torch.cat([other.posterior_std_devs, self.posterior_std_devs]))

    def slice(self, start, end):
        return StatePredictions(beliefs=self.beliefs[start, end],
                                prior_states=self.prior_states[start, end],
                                prior_means=self.prior_means[start, end],
                                prior_std_devs=self.prior_std_devs[start, end],
                                posterior_states=self.posterior_states[start, end],
                                posterior_means=self.posterior_means[start, end],
                                posterior_std_devs=self.posterior_std_devs[start, end])


class ImaginedTrajectories(NamedTuple):
    beliefs: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    rewards: Optional[torch.Tensor]
    logprobs: Optional[torch.Tensor]

    def detach(self):
        return ImaginedTrajectories(
            beliefs=self.beliefs.detach(),
            states=self.states.detach(),
            actions=self.actions.detach(),
            rewards=self.rewards.detach() if type(self.rewards) is torch.Tensor else self.rewards,
            logprobs=self.logprobs.detach() if type(self.logprobs) is torch.Tensor else self.logprobs
        )


class Cell(NamedTuple):
    x: int
    y: int
