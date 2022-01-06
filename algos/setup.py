from stable_baselines3 import A2C

from algos.PlaNet.planet import PlaNet
from algos.PlaNet.policies import DiscreteMPCPlanner
from algos.PlaNet.world_model import DreamerModel
import os
from buffers.chunk_buffer import ChunkReplayBuffer
from buffers.introspective_buffer import IntrospectiveChunkReplayBuffer
from algos.dreamer.dreamer import Dreamer
from algos.dreamer.policies import DreamerPolicy

__all__ = ['setup_dreamer', 'setup_a2c', 'setup_planet']


def _update_after(cfg):
    return 'step' if 'update_after' in cfg.training and cfg.training.update_after == 'step' else 'episode'


def _get_step_size(cfg, env, update_after):
    from utils import total_timesteps
    if 'lr_steps' not in cfg.mpc:
        return 10000 if update_after == 'step' else 100
    ignore_exploration = 'reset_value' in cfg.exploration and cfg.exploration.reset_value
    if update_after == 'step':
        return total_timesteps(cfg, False, not ignore_exploration) // cfg.mpc.lr_steps
    return total_timesteps(cfg, False, not ignore_exploration) // env.unwrapped.max_steps // cfg.mpc.lr_steps


def setup_planet(cfg, model_params, env, device):
    update_after = _update_after(cfg)
    lr_step_size = _get_step_size(cfg, env, update_after)
    world_model = DreamerModel.from_args(model_params, device)
    policy = DiscreteMPCPlanner.from_args(cfg.mpc, env, world_model, lr_step_size=lr_step_size)
    return PlaNet(policy, env,
                  learning_starts=cfg.training.warmup_steps,
                  gradient_steps=cfg.training.gradient_steps, resample_batch=cfg.training.resample_batch,
                  verbose=1, tensorboard_log=os.path.join(os.getcwd(), 'tensorboard'), seed=cfg.environment.seed,
                  device=device, train_freq=(cfg.training.update_every, update_after),
                  exploration_min=cfg.random_action.exploration_min,
                  exploration_prob=cfg.random_action.exploration_prob,
                  exploration_decay_steps=cfg.random_action.exploration_decay_steps,
                  replay_buffer_class=IntrospectiveChunkReplayBuffer if cfg.training.introspect_buffer else ChunkReplayBuffer,
                  replay_buffer_kwargs={'chunks_can_cross_episodes': cfg.training.sample_across_episodes})


def setup_dreamer(cfg, model_params, env, device):
    update_after = _update_after(cfg)
    lr_step_size = _get_step_size(cfg, env, update_after)
    world_model = DreamerModel.from_args(model_params, device)
    policy = DreamerPolicy.from_config(cfg.dreamer, model_params, env, world_model, lr_step_size=lr_step_size)
    return Dreamer(policy, env,
                   learning_starts=cfg.training.warmup_steps,
                   gradient_steps=cfg.training.gradient_steps, resample_batch=cfg.training.resample_batch,
                   verbose=1, tensorboard_log=os.path.join(os.getcwd(), 'tensorboard'), seed=cfg.environment.seed,
                   device=device, train_freq=(1, update_after),
                   exploration_min=cfg.random_action.exploration_min,
                   exploration_prob=cfg.random_action.exploration_prob,
                   exploration_decay_steps=cfg.random_action.exploration_decay_steps,
                   replay_buffer_kwargs={'chunks_can_cross_episodes': cfg.training.sample_across_episodes})


def setup_a2c(cfg, model_params, env, device):
    return A2C('MlpPolicy', env, verbose=1, tensorboard_log=os.path.join(os.getcwd(), 'tensorboard'),
               seed=cfg.environment.seed, device=device)
