import abc
from abc import ABC


class Optimizer(ABC):

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def has_value_model(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def train_batch(self, states, embeddings, actions, rewards, dones, gamma, logger):
        pass

    def before_updates(self):
        pass
