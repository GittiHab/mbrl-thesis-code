from collections import defaultdict
from typing import Optional, Tuple, Dict, Any

from stable_baselines3.common.noise import ActionNoise

from algos.PlaNet.optimizer import Optimizer
from type_aliases import StatePredictions, ExtendedReplayBufferSamples
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from algos.PlaNet.policies import MPCPlanner
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import ReplayBufferSamples, MaybeCallback, GymEnv, TrainFreq, \
    TrainFrequencyUnit
from models.models import bottle
from torch.distributions import Normal
from torch.distributions.independent import Independent

from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from stable_baselines3.common.callbacks import BaseCallback
from buffers.chunk_buffer import ChunkReplayBuffer
from callbacks import LogEpisodeReward


class ResetEnvCallback(BaseCallback):

    def __init__(self, policy: MPCPlanner, verbose=0):
        super().__init__(verbose)
        self.policy: MPCPlanner = policy

    def _on_step(self) -> bool:
        return True

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        super().update_locals(locals_)
        if locals_['done']:
            self.policy.reset_state()


class PlaNet(OffPolicyAlgorithm):
    def __init__(self, policy, env, chunk_size=5, learning_starts=100, gamma=0.99, train_freq=(1, "episode"),
                 value_optimizers=[], gradient_steps=50, resample_batch=False,
                 tensorboard_log=None, verbose=0, device="auto", seed=None, _init_setup_model=True,
                 free_nats=3, beta=1.0, reward_scale=5.0, pcont_scale=5.0, grad_clip_norm=100.0,
                 exploration_prob=0., exploration_decay_steps=None, exploration_min=0., replay_buffer_class=None,
                 replay_buffer_kwargs=None):
        super().__init__(
            policy,
            env,
            None,  # TODO: register policy_base
            learning_rate=1e-4,  # dummy: LRs are managed by the components (because we have two: world and value model)
            buffer_size=1000000,
            learning_starts=learning_starts,
            batch_size=256,
            tau=0.005,
            gamma=gamma,
            train_freq=TrainFreq(frequency=train_freq[0], unit=TrainFrequencyUnit(train_freq[1])),
            gradient_steps=gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=False,
            policy_kwargs=None,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=False,
            create_eval_env=False,
            monitor_wrapper=True,
            seed=seed,
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            sde_support=True,
            remove_time_limit_termination=False,
            supported_action_spaces=None
        )
        self.resample_batch = resample_batch

        self.free_nats = free_nats
        self.beta = beta
        self.reward_scale = reward_scale
        self.pcont_scale = pcont_scale
        self.grad_clip_norm = grad_clip_norm

        self.exploration_prob = exploration_prob
        self.exploration_decay_steps = exploration_decay_steps if exploration_decay_steps is not None \
            else learning_starts * 2
        self.exploration_min = exploration_min
        if self.exploration_prob == 0.:
            self.exploration_decay = 0.
            self.exploration_decay_steps = 0
        else:
            self.exploration_decay = -float(
                self.exploration_decay_steps * np.log(2) / (np.log(exploration_min) - np.log(exploration_prob)))

        self.policy = self.policy_class
        self.chunk_size = chunk_size
        self.optimizers = {}
        [self.add_value_optimizer(optimizer) for optimizer in value_optimizers]

        if _init_setup_model:
            self._setup_optimizers()
            self._setup()

    def add_value_optimizer(self, value_optimizer: Optimizer):
        self.optimizers[value_optimizer.name] = value_optimizer

    def remove_value_optimizer(self, value_optimizer):
        del self.optimizers[value_optimizer.name]

    def _setup_model(self):
        return self._setup()

    def _setup_optimizers(self):
        self.add_value_optimizer(self.policy.reward_space)

    def _setup(self):
        # self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use DictReplayBuffer if needed
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, gym.spaces.Dict):
                # self.replay_buffer_class = DictReplayBuffer
                raise NotImplementedError('Dict observation spaces are not yet supported by default. '
                                          'Please pass a replay buffer supporting both dicts and sampling chunks.')
            else:
                self.replay_buffer_class = ChunkReplayBuffer

        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.chunk_size,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

    def _build_callback(self, callback, log_interval):
        callbacks = []
        if log_interval == 1:
            callbacks.append(LogEpisodeReward(self.logger))
        if callback is not None:
            callbacks.append(callback)
        callbacks.append(ResetEnvCallback(self.policy))
        return callbacks

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "run",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OffPolicyAlgorithm":

        return super().learn(total_timesteps,
                             self._build_callback(callback, log_interval),
                             log_interval,
                             eval_env,
                             eval_freq,
                             n_eval_episodes,
                             tb_log_name,
                             eval_log_path,
                             reset_num_timesteps)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # TODO: how do we handle the recurrency?
        return super().predict(observation, state, mask, deterministic)

    @property
    def exploration_amount(self):
        # TODO: change calculation s.t. it depends only on the number time steps
        #       (not updates as this may change with episode length).
        amount = self.exploration_prob
        if self.exploration_decay_steps:
            amount *= 0.5 ** (
                    (self._n_updates / self.gradient_steps - 1) / self.exploration_decay)
        return max(self.exploration_min, amount)

    def _sample_action(
            self, learning_starts: int, action_noise: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        action, buffer_action = super()._sample_action(learning_starts, action_noise)
        # SB3 does not add exploration in discrete case:
        # TODO we could use this here to return buffer_action one-hot encoded and action as the argmax...
        #      (there is a related to-do somewhere else).
        if self._discrete_env:
            if self.exploration_amount > 0 and np.random.rand() < self.exploration_amount:
                action = buffer_action = np.array([self.action_space.sample()])
        return action, buffer_action

    @property
    def _discrete_env(self):
        return hasattr(self.env.action_space, 'n')

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # Based on the DQN implementation in stable-baselines 3 and the dreamer-pytorch project by zhaoyi11
        # https://github.com/DLR-RM/stable-baselines3/blob/3efab0d267e74cb03264411d4500ddde0c163404/stable_baselines3/dqn/dqn.py#L154
        # https://github.com/zhaoyi11/dreamer-pytorch

        # Update learning rate according to schedule
        # self._update_learning_rate(self.policy.optimizer)

        losses_obs = []
        losses_reward = []
        losses_kl = []
        losses_pcont = []
        losses_value = defaultdict(list)
        # Sample replay buffer
        if not self.resample_batch:
            replay_data = self.replay_buffer.sample(batch_size,
                                                    env=self._vec_normalize_env)  # type: ReplayBufferSamples
            replay_data: ExtendedReplayBufferSamples = ExtendedReplayBufferSamples.from_replay_data(replay_data)

        [optimizer.before_updates() for optimizer in self.optimizers.values()]
        for _ in range(gradient_steps):
            if self.resample_batch:
                replay_data = self.replay_buffer.sample(batch_size,
                                                        env=self._vec_normalize_env)  # type: ReplayBufferSamples
                replay_data: ExtendedReplayBufferSamples = ExtendedReplayBufferSamples.from_replay_data(replay_data)
            # TODO: solve this more nicely:
            actions = one_hot(replay_data.actions, num_classes=self.action_space.n).squeeze(
                -2) if hasattr(self.action_space, 'n') else replay_data.actions
            states, embeddings, losses = \
                self._update_world_model(replay_data.all_observations.type(torch.float),
                                         actions,
                                         replay_data.rewards,
                                         replay_data.dones,
                                         batch_size)

            for optimizer in self.optimizers.values():
                loss = optimizer.train_batch(states, embeddings, actions, replay_data.rewards,
                                             replay_data.dones, self.gamma, self.logger)
                losses_value[optimizer.name].append(loss)

            observation_loss, reward_loss, kl_loss, pcont_loss = losses
            losses_obs.append(observation_loss)
            losses_reward.append(reward_loss)
            losses_kl.append(kl_loss)
            losses_pcont.append(pcont_loss)

        [optimizer.after_updates(self.logger) for optimizer in self.optimizers.values()]

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/observation_loss", np.mean(losses_obs))
        self.logger.record("train/reward_loss", np.mean(losses_reward))
        self.logger.record("train/kl_loss", np.mean(losses_kl))
        self.logger.record("train/p_cont_loss", np.mean(losses_pcont))
        if self._discrete_env:
            self.logger.record("train/exploration_amount", self.exploration_amount)
        for name, losses in losses_value.items():
            self.logger.record("train/value_loss_" + name, np.mean(losses))

    def _update_world_model(self, observations, actions, rewards, dones, batch_size):
        # This method is a modified version from dreamer-pytorch by zhaoyi11 published under the MIT license.
        # https://github.com/zhaoyi11/dreamer-pytorch/blob/500ae4fcd7143342b1eb77835e39c5d1d4986170/agent.py#L142

        nonterminals = torch.logical_not(dones)

        # get state and belief of samples
        init_belief = torch.zeros(batch_size, self.policy.world_model.belief_size, device=observations.device)
        init_state = torch.zeros(batch_size, self.policy.world_model.state_size, device=observations.device)
        init_action = torch.zeros(batch_size, self.policy.world_model.action_size, device=observations.device)

        embeddings = bottle(self.policy.world_model.encoder, (observations.type(torch.float),))

        # Update belief/state using posterior from previous belief/state, previous action and current observation
        # (over entire sequence at once)
        s0_predictions: StatePredictions = self.policy.world_model.infer_state(
            observations[0], init_action, init_belief, init_state, True)
        state_predictions: StatePredictions = self.policy.world_model.transition_model(
            s0_predictions.posterior_states.squeeze(0),
            actions,
            s0_predictions.beliefs.squeeze(0),
            embeddings[1:],
            nonterminals)
        state_predictions = state_predictions.prepend(s0_predictions)

        # update params of world model
        world_model_loss = self._compute_loss_world(
            state=state_predictions,
            data=(observations, rewards, nonterminals)
        )
        observation_loss, reward_loss, kl_loss, pcont_loss = world_model_loss
        self.policy.world_model.world_optimizer.zero_grad()
        (observation_loss + reward_loss + kl_loss + pcont_loss).backward()
        nn.utils.clip_grad_norm_(self.policy.world_model.world_param, self.grad_clip_norm,
                                 norm_type=2)
        self.policy.world_model.world_optimizer.step()
        losses = (
            observation_loss.item(),
            reward_loss.item(),
            kl_loss.item(),
            pcont_loss.item() if isinstance(pcont_loss, torch.Tensor) else 0.)

        return state_predictions, embeddings, losses

    def _compute_loss_world(self, state, data):
        # unpackage data
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = \
            state
        observations, rewards, nonterminals = data

        observation_loss = F.mse_loss(
            bottle(self.policy.world_model.observation_model, (beliefs, posterior_states)),
            observations,
            reduction='none').sum(dim=2).mean(dim=(0, 1))

        reward_loss = F.mse_loss(
            bottle(self.policy.world_model.reward_model, (beliefs[1:], posterior_states[1:])),
            rewards.squeeze(-1),
            reduction='none').mean(dim=(0, 1))

        # transition loss
        kl_loss = torch.maximum(
            kl_divergence(
                Independent(Normal(posterior_means, posterior_std_devs), 1),
                Independent(Normal(prior_means, prior_std_devs), 1)),
            torch.tensor(self.free_nats, device=posterior_means.device)).mean(dim=(0, 1))

        pcont_loss = F.binary_cross_entropy(
            bottle(self.policy.world_model.pcont_model, (beliefs[1:], posterior_states[1:])),
            nonterminals.type(torch.float).squeeze()) if self.policy.world_model.pcont else 0.

        return observation_loss, self.reward_scale * reward_loss, self.beta * kl_loss, (
                self.pcont_scale * pcont_loss)
