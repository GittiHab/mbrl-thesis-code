'''
Different functions used to estimate the value of states.
These estimates are used as targets for training the value network.
'''

from enum import Enum

import torch
from models.models import bottle
from torch.nn.functional import mse_loss

__all__ = ['one_step_td', 'DreamerValueTarget', 'ValueTarget', 'get_target_function', 'Targets']


class Targets(Enum):
    OneStep = 'one_step'
    Dreamer = 'dreamer'


def get_target_function(config):
    if 'target_fn' not in config:
        return one_step_td
    target = Targets(config.target_fn.lower())
    if target == Targets.Dreamer:
        return DreamerValueTarget(config.lambda_discount)
    elif target == Targets.OneStep:
        return one_step_td
    raise NotImplementedError('Given target function is not implemented.')


def one_step_td(rewards, dones, gamma, values_next):
    return rewards + (~dones) * gamma * values_next


class ValueTarget:
    def __call__(self, rewards, dones, gamma, values_next):
        raise NotImplementedError()


class DreamerValueTarget(ValueTarget):

    def __init__(self, lambda_discount):
        self.lambda_discount = lambda_discount

    @staticmethod
    def _weighted_return(reward, value_next, pcont, lambda_):
        # This method was taken from dreamer-pytorch by zhaoyi11
        # https://github.com/zhaoyi11/dreamer-pytorch/blob/500ae4fcd7143342b1eb77835e39c5d1d4986170/agent.py#L16
        # Code modified.
        """
        Calculate the target value, following equation (5-6) in Dreamer
        :param reward, value_next: imagined rewards and values, dim=[horizon, (chuck-1)*batch, reward/value_shape]
        :param bootstrap: the last predicted value, dim=[(chuck-1)*batch, 1(value_dim)]
        :param pcont: gamma
        :param lambda_: lambda
        :return: the target value, dim=[horizon, (chuck-1)*batch, value_shape]
        """
        assert list(reward.shape) == list(value_next.shape), "The shape of reward and value should be similar"
        if isinstance(pcont, (int, float)):
            pcont = pcont * torch.ones_like(reward)

        inputs = reward + pcont * value_next * (1 - lambda_)  # dim=[horizon, (chuck-1)*B, 1]
        outputs = []
        last = value_next[-1]

        for t in reversed(range(reward.shape[0])):  # for t in horizon
            inp = inputs[t]
            last = inp + pcont[t] * lambda_ * last
            outputs.append(last)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns

    def __call__(self, rewards, dones, gamma, values_next):
        # This method is based on code from dreamer-pytorch by zhaoyi11
        # https://github.com/zhaoyi11/dreamer-pytorch/blob/500ae4fcd7143342b1eb77835e39c5d1d4986170/agent.py#L190

        # rewards, values_next, gamma, lambda_discount

        # with torch.no_grad():
        #     # calculate the target with the target nn
        #     target_values = bottle(self.target_value_model, (beliefs, states))
        pcont = gamma * torch.ones_like(rewards)

        values_next[dones[1:] == 1] = 0.
        pcont[dones == 1] = 0.

        returns = self._weighted_return(rewards[:-1], values_next, pcont[1:], self.lambda_discount)
        return returns
        #
        # value_pred = bottle(self.value_model, (beliefs, states))
        #
        # value_loss = mse_loss(value_pred[:-1], target_return[1:], reduction="none")
        #
        # return value_loss.mean(dim=(0, 1))
