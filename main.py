import sys

from gym_minigrid.wrappers import *
from stable_baselines3 import A2C

from algos.PlaNet.planet import PlaNet
from algos.PlaNet.reward_space import ExplorationRewardSpace
from checkpointing import Checkpointing, NullCheckpointing
from exploration.base import STDExplorer
from utils import get_git_revision_hash, get_git_branch
from wrappers import SymbolicWrapper, SimpleActionsWrapper, StepwiseRewardWrapper
from hydra.utils import to_absolute_path
from algos.setup import *
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from models.model_params import ModelParams
import os.path
from debug import manual_eval

log = logging.getLogger(__name__)


def build_env(config):
    env = gym.make(config.env)
    env_seed = None
    if 'env_seed' in config and config.env_seed is not None:
        env_seed = config.env_seed
        env = ReseedWrapper(env, [env_seed])
    env = SymbolicWrapper(env, directions=False)
    env = SimpleActionsWrapper(env)
    env = StepwiseRewardWrapper(env, muted=config.muted if 'muted' in config else False)
    return env, env_seed


def build_agent(cfg, model_params, env, device):
    algo = cfg.algorithm.lower()
    if algo == 'planet':
        setup_method = setup_planet
    elif algo == 'a2c':
        setup_method = setup_a2c
    elif algo == 'dreamer':
        setup_method = setup_dreamer
    else:
        raise NotImplementedError('The passed algorithm {} is not implemented'.format(cfg.algorithm))
    return setup_method(cfg, model_params, env, device)


def load_agent(algorithm, path, env, device):
    algo = algorithm.lower()
    if algo == 'planet':
        cls = PlaNet
    elif algo == 'a2c':
        cls = A2C
    else:
        raise NotImplementedError('The passed algorithm {} is not implemented'.format(algorithm))
    return cls.load(path, env=env, device=device)


def explore(model: PlaNet, cfg, model_params, callback):
    explorer = STDExplorer.from_config(1, cfg.exploration, model_params)
    exploration_reward = ExplorationRewardSpace(explorer,
                                                value_lr=cfg.mpc.value_lr,
                                                value_layers=cfg.mpc.value_layers if 'value_layers' in cfg.mpc else 3,
                                                # state_size=model_params.state_size,
                                                belief_size=model_params.belief_size,
                                                hidden_size=cfg.mpc.hidden_size if 'hidden_size' in cfg.mpc else model_params.hidden_size,
                                                lr_strategy=None, # No learning rate schedule for exploration phase
                                                device=model_params.device,
                                                name='value_exploration',
                                                reset_target_every=cfg.exploration.reset_target_every,
                                                update_target_every=cfg.mpc.update_target_every if 'update_target_every' in cfg.mpc else None)
    model.add_value_optimizer(exploration_reward)

    real_reward = model.policy.reward_space
    model.policy.set_reward_space(exploration_reward)

    model.learn(total_timesteps=(model.learning_starts + cfg.exploration.steps), callback=callback,
                tb_log_name='explore', log_interval=cfg.training.log_interval)
    model.learning_starts = 0

    model.policy.set_reward_space(real_reward)
    if cfg.exploration.reset_value:
        real_reward.reset_value_params()
    model.remove_value_optimizer(exploration_reward)


def evaluate(env, model, config):
    obs = env.reset()
    done = False
    rewards = []
    episode = 0
    for i in range(config.max_timesteps):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if 'render' in config and config.render:
            env.render()
        rewards.append(reward)
        if done:
            obs = env.reset()
            if hasattr(model.policy, 'reset_state'):
                model.policy.reset_state()
            episode += 1
            if episode == config.episodes:
                break
    mean_return = np.sum(rewards) / episode
    return mean_return, episode


def exploration_phase(cfg):
    return 'exploration' in cfg and 'reward_type' in cfg.exploration


def _timesteps(model, timesteps):
    if hasattr(model, 'learning_starts'):
        return model.learning_starts + timesteps
    return timesteps


def do_train(cfg, model, model_params):
    checkpointing = NullCheckpointing()
    if 'checkpointing' in cfg:
        checkpointing = Checkpointing.from_config(cfg.checkpointing)

    if exploration_phase(cfg):
        explore(model, cfg, model_params, checkpointing.get_exploration_callback())

    timesteps = _timesteps(model, cfg.environment.steps)
    if timesteps > 0:
        model.learn(total_timesteps=timesteps, callback=checkpointing.get_exploitation_callback(), tb_log_name='run',
                    log_interval=cfg.training.log_interval)

    if hasattr(model, 'save_replay_buffer'):
        model.save_replay_buffer(os.path.join('.', 'replay_buffer'))
    model.save(os.path.join('.', 'model'))


@hydra.main(config_path="experiments", config_name="base")
def main(cfg: DictConfig):
    log.info('command line arguments: ' + ' '.join(sys.argv[1:]))
    log.info('commit: ' + str(get_git_revision_hash()) + ' on branch ' + str(get_git_branch()))
    log.info(OmegaConf.to_yaml(cfg))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    env, env_seed = build_env(cfg.environment)
    model_params = ModelParams.from_config(cfg.model_params, env, device)
    train = True
    if 'load' in cfg:
        train = False
        if cfg.load == '':
            cfg.load = input('Path to model: ')
        model = load_agent(cfg.algorithm, to_absolute_path(cfg.load.strip()), env, device)
    else:
        model = build_agent(cfg, model_params, env, device)

    log.info('env:      ' + str(env.spec.id))
    log.info('env_seed: ' + str(env_seed))
    log.info('seed:     ' + str(model.seed))

    if train:
        do_train(cfg, model, model_params)

    if 'manual_eval' not in cfg or not cfg.manual_eval:
        mean_return, episodes = evaluate(env, model, cfg.evaluation)
        print(mean_return)
        print('Finished evaluation episodes:', episodes)
        log.info('Evaluation reward: ' + str(mean_return))
    else:
        manual_eval(env, model, cfg.evaluation)

if __name__ == '__main__':
    main()
