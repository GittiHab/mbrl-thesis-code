from dataclasses import dataclass
from typing import Union
import torch

__all__ = ['ModelParams']


@dataclass
class ModelParams:
    observation_size: int
    belief_size: int
    state_size: int
    action_size: int
    hidden_size: int
    embedding_size: int
    dense_act: str
    cnn_act: str
    pcont: bool
    world_lr: float
    symbolic: bool
    device: Union[str, torch.device]

    @staticmethod
    def from_config(config, env, device):
        action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        return ModelParams(env.observation_space.shape[0],
                           config.belief_size,
                           config.state_size,
                           action_size,
                           config.hidden_size,
                           config.embedding_size,
                           config.dense_act,
                           config.cnn_act,
                           config.pcont,
                           config.world_lr,
                           config.symbolic,
                           device)
