from abc import ABC
from models.ensemble import STDEnsemble
from models.models import RewardModel
from models.models import bottle

from models.mlp import MLP
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

__all__ = ['Explorer', 'NullExplorer', 'STDExplorer', 'StateSTDExplorer', 'ObservationSTDExplorer', 'RewardSTDExplorer',
           'RewardModelProxy', 'OneStepPredictor', 'OneStepPredictorYusukeurakami']


class Explorer(ABC):
    def __call__(self, reward, beliefs, states, actions):
        raise NotImplementedError()

    def exploration_reward(self, belief, state, action):
        raise NotImplementedError()

    def train_batch(self, *data):
        raise NotImplementedError()

    def save(self, filepath):
        raise NotImplementedError()

    def load(self, filepath):
        raise NotImplementedError()


class NullExplorer(Explorer):
    def __call__(self, reward, beliefs, states, actions):
        return reward

    def exploration_reward(self, belief, state, action):
        return 0.

    def train_batch(self, *data):
        return 0

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass


class STDExplorer(Explorer):
    _weight = 0

    @staticmethod
    def from_config(weight, config, model_params):
        reward_type = config.reward_type
        if reward_type == 'state':
            type = StateSTDExplorer
        elif reward_type == 'observation':
            type = ObservationSTDExplorer
        elif reward_type == 'reward':
            type = RewardSTDExplorer
        else:
            raise KeyError(
                'The exploration reward type {} does not exist. Please use "state" or "reward"'.format(reward_type))

        return type(weight,
                    belief_size=model_params.belief_size,
                    state_size=model_params.state_size,
                    hidden_size=model_params.hidden_size,
                    embedding_size=model_params.embedding_size,
                    action_size=model_params.action_size,
                    device=model_params.device,
                    learning_rate=config.learning_rate,
                    scale=config.scale,
                    input_=config.input)

    def __init__(self,
                 weight,
                 belief_size,
                 state_size,
                 hidden_size,
                 embedding_size,
                 action_size,
                 device='cpu',
                 instances=5,
                 learning_rate=8e-5,
                 scale=10000,
                 input_='belief'):
        super().__init__()
        self.input = input_
        self._ensemble = self._setup_model(belief_size, state_size, action_size, hidden_size, embedding_size, instances,
                                           device)
        self.weight(weight)
        self._scale = scale
        self._optimizer = optim.Adam(self._ensemble.parameters(), lr=learning_rate)

    def _setup_model(self, belief_size, state_size, action_size, hidden_size, embedding_size, instances, device):
        raise NotImplementedError()

    def create_default_model(self, belief_size, state_size, action_size, hidden_size, output_size, instances, device):
        input_size = belief_size if self.input == 'belief' else state_size
        # return STDEnsemble(OneStepPredictorYusukeurakami,
        #                    (input_size, action_size, hidden_size, output_size), instances).to(device)
        return STDEnsemble(OneStepPredictor, (input_size, action_size, hidden_size, output_size), instances).to(device)

    def save(self, filepath):
        data = {'optimizer': self._optimizer.state_dict(), 'ensemble': self._ensemble.save()}
        torch.save(data, filepath)

    def load(self, filepath):
        data = torch.load(filepath)
        self._optimizer.load_state_dict(data['optimizer'])
        self._ensemble.load(data=data['ensemble'])

    def weight(self, weight=None):
        if weight is None:
            return self._weight
        assert 0 <= weight <= 1, 'Weight needs to be between or equal to 0 and 1 but was {}'.format(weight)
        self._weight = weight

    def __call__(self, reward, beliefs, states, actions):
        exploration_reward = self.exploration_reward(beliefs, states, actions)
        if self._weight == 1:
            return exploration_reward
        return (1 - self._weight) * reward + self._weight * exploration_reward

    def exploration_reward(self, belief, state, action):
        return self._ensemble(*self.input_from(belief, state, action)) * self._scale

    def input_from(self, belief, state, action=None):
        if self.input == 'state':
            input_ = [state]
        elif self.input == 'separate':
            input_ = [belief, state]
        elif self.input == 'both':
            input_ = [torch.cat([belief, state], dim=-1)]
        elif self.input == 'belief':
            input_ = [belief]
        else:
            raise NotImplementedError('Input type {} is not supported.'.format(self.input))
        return input_ if action is None else tuple(input_ + [action])

    def train_batch(self, rewards, beliefs, states, actions, embedding):
        # TODO: make this nicer (that's why I am accessing a protected member here!)
        total_loss = None
        for model in self._ensemble._models:
            loss = self._loss(model, rewards, beliefs, states, actions, embedding)
            total_loss = loss if total_loss is None else total_loss + loss
        self.optimize(total_loss)
        return total_loss.item()

    def _loss(self, model, rewards, beliefs, states, actions, embedding):
        raise NotImplementedError()

    def optimize(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


class RewardSTDExplorer(STDExplorer):
    def __init__(self, *args, **kwargs):
        if 'input_' in kwargs:
            kwargs['input_'] = 'separate'
        super().__init__(*args, **kwargs)

    def _setup_model(self, belief_size, state_size, action_size, hidden_size, embedding_size, instances, device):
        return STDEnsemble(RewardModelProxy,
                           (belief_size, state_size, hidden_size),
                           instances).to(device)

    def _loss(self, model, rewards, beliefs, states, actions, embedding):
        return F.mse_loss(
            bottle(model, (beliefs.detach(), states.detach())),
            rewards.detach(),
            reduction='mean')


class ObservationSTDExplorer(STDExplorer):
    def _setup_model(self, belief_size, state_size, action_size, hidden_size, embedding_size, instances, device):
        return self.create_default_model(belief_size, state_size, action_size, hidden_size, embedding_size, instances,
                                         device)

    def _loss(self, model, rewards, beliefs, states, actions, embedding):
        input_ = self.input_from(beliefs, states)[0]
        return F.mse_loss(
            bottle(model, (input_[:-1].detach(), actions.detach())), embedding[1:].detach(),
            reduction='none').sum(dim=-1).mean()


class StateSTDExplorer(STDExplorer):
    def _setup_model(self, belief_size, state_size, action_size, hidden_size, embedding_size, instances, device):
        return self.create_default_model(belief_size, state_size, action_size, hidden_size, state_size, instances,
                                         device)

    def _loss(self, model, rewards, beliefs, states, actions, embedding):
        input_ = self.input_from(beliefs, states)[0]
        return F.mse_loss(
            bottle(model, (input_[:-1].detach(), actions.detach())), states[1:].detach(),
            reduction='none').sum(dim=-1).mean()


class RewardModelProxy(RewardModel):

    def forward(self, *params):
        if len(params) == 2:
            belief, state = params
        else:
            belief, state, action = params
        return super().forward(belief, state)


class OneStepPredictor(nn.Module):
    """
    This is based on the description of the latent disagreement model in the Plan2Explore paper [1].

    [1] Sekar, R. et al. “Planning to Explore via Self-Supervised World Models.” ArXiv abs/2005.05960 (2020)
    """

    def __init__(self, belief_size, action_size, hidden_size, embedding_size, layers=2):
        super().__init__()
        self._mlp = MLP(layers, belief_size + action_size, hidden_size, embedding_size)

    def forward(self, *params):
        if len(params) == 2:
            beliefs, actions = params
        else:
            beliefs, states, actions = params
        return self._mlp(torch.cat([beliefs, actions], dim=-1))  # TODO: p2e impl. seems to use distribution here(?)


class OneStepPredictorYusukeurakami(nn.Module):
    """
    This is from [yusukeurakami] from the plan2explore-pytorch repository.
    https://github.com/yusukeurakami/plan2explore-pytorch/blob/63336f2a061e6b8a6aecf5d287927804a4f5be60/models.py
    """

    def __init__(self, state_size, action_size, hidden_size, embedding_size, activation_function='relu',
                 model_width_factor=1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size * model_width_factor  # TODO: do we want to change this? Try it some point.
        self.fc1 = nn.Linear(state_size + action_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + action_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + action_size, embedding_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, *params):
        if len(params) == 2:
            belief, action = params
        else:
            belief, state, action = params
        belief = belief.detach()
        action = action.detach()
        x = torch.cat([belief, action], dim=-1)  # torch.Size([49, 50, 30]) torch.Size([49, 50, 6])
        hidden = self.act_fn(self.fc1(x))
        hidden = torch.cat([hidden, action], dim=-1)
        hidden = self.act_fn(self.fc2(hidden))
        hidden = torch.cat([hidden, action], dim=-1)
        mean = self.fc3(hidden)
        return mean
