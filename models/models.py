# This document is a modified version of the original document from dreamer-pytorch by zhaoyi11
# published under the MIT license.
# https://github.com/zhaoyi11/dreamer-pytorch/blob/500ae4fcd7143342b1eb77835e39c5d1d4986170/models.py
from typing import Optional, List, Union

import numpy as np

from type_aliases import StatePredictions
import torch
import torch.distributions
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import one_hot


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    if type(y) is tuple:
        y = y[1]
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.modules = [self.fc_embed_state_action, self.fc_embed_belief_prior, self.fc_state_prior,
                        self.fc_embed_belief_posterior, self.fc_state_posterior]

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, prev_belief: torch.Tensor,
                observations: Optional[torch.Tensor] = None, nonterminals: Optional[torch.Tensor] = None,
                prior_params: Optional[bool] = None, return_start=False) -> Union[StatePredictions, List[torch.Tensor]]:
        """
        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        :param return_start: When only imagining next states, also return the initial state and belief. Is ignored in other cases.
        :return:
        """
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states = [torch.empty(0)] * T, [torch.empty(0)] * T
        prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T
        posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [
            torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[
                t]  # Select appropriate previous state
            # Mask if previous transition was terminal
            _state = _state if (nonterminals is None or t == 0) else _state * nonterminals[t - 1]
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t] if (nonterminals is None or t == 0) else beliefs[t] * nonterminals[t - 1])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1])
        # Return new hidden states
        s = 1 if not return_start or observations is not None or prior_params else 0
        hidden = [torch.stack(beliefs[s:], dim=0), torch.stack(prior_states[s:], dim=0)]
        if prior_params or prior_params is None and observations is not None:
            hidden += [torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0),
                       torch.stack(posterior_std_devs[1:], dim=0)]
            hidden = StatePredictions(*hidden)
        return hidden


class SymbolicObservationModel(nn.Module):
    def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)

    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
    else:
        raise NotImplementedError()


class RewardModel(nn.Module):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden)
        reward = reward.squeeze(dim=-1)
        return reward


class SymbolicEncoder(nn.Module):
    def __init__(self, observation_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)

    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        raise NotImplementedError()


class PCONTModel(nn.Module):
    """ predict the prob of whether a state is a non-terminal state. """

    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        x = self.fc4(hidden).squeeze(dim=1)
        p = torch.sigmoid(x)
        return p


class ActorModel(nn.Module):
    def __init__(self,
                 action_size,
                 belief_size,
                 state_size,
                 hidden_size,
                 mean_scale=5,
                 min_std=1e-4,
                 init_std=5,
                 activation_function="elu",
                 discrete=False):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        out_size = 2 * action_size if not discrete else action_size
        self.fc5 = nn.Linear(hidden_size, out_size)
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.discrete = discrete

    def forward(self, belief, state, deterministic=False, with_logprob=False, probs=False):
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=-1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        if self.discrete:
            hidden = self.fc5(hidden)
            dist = CategoricalDist(logits=hidden)
        else:
            hidden = self.fc5(hidden)
            mean, std = torch.chunk(hidden, 2, dim=-1)
            mean = self.mean_scale * torch.tanh(
                mean / self.mean_scale)  # bound the action to [-5, 5] --> to avoid numerical instabilities.  For computing log-probabilities, we need to invert the tanh and this becomes difficult in highly saturated regions.
            std = F.softplus(std + raw_init_std) + self.min_std
            dist = torch.distributions.Normal(mean, std)
            transform = [torch.distributions.transforms.TanhTransform()]
            dist = torch.distributions.TransformedDistribution(dist, transform)
            dist = torch.distributions.independent.Independent(dist,
                                                               1)  # Introduces dependence between actions dimension
            dist = SampleDist(
                dist)  # because after transform a distribution, some methods may become invalid, such as entropy, mean and mode, we need SmapleDist to approximate it.

        if probs:
            return dist.probs

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        if with_logprob:
            logp_pi = dist.log_prob(action)
        else:
            logp_pi = None

        return action, logp_pi


class SampleDist:
    """
    After TransformedDistribution, many methods becomes invalid, therefore, we need to approximate them.
    """

    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        # print("dist in mode", sample.shape)
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)


class CategoricalDist:
    # Modified from https://github.com/danijar/dreamer/blob/0256080fb213d9da6d7b706f5a9afc54ba686734/tools.py#L250
    def __init__(self, logits=None, probs=None):
        self._dist = torch.distributions.OneHotCategorical(logits=logits, probs=probs)
        self._num_classes = logits.size(-1) if logits is not None else probs.size(-1)
        self.dtype = torch.float32

    @property
    def name(self):
        return 'CategoricalDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def log_prob(self, samples):
        return self._dist.log_prob(samples)

    def prob(self, samples):
        # TODO: can this be done nicer???
        # with something like self._dist.probs[np.arange(samples.size(0)), samples.squeeze(0)]?
        return torch.exp(self.log_prob(samples))

    @property
    def mean(self):
        return self.mode  # self._dist.probs

    @property
    def mode(self):
        return one_hot(self._dist.probs.argmax(-1), self._num_classes).type(self.dtype)

    def rsample(self):
        samples = self._dist.sample().type(self.dtype)
        samples = self._dist.probs + (samples - self._dist.probs).detach()
        return samples
