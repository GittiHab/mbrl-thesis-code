import warnings
from typing import Union, Optional, List, Dict, Any

import numpy as np
import torch
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class ChunkReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            chunk_size: int,
            device: Union[torch.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            chunks_can_cross_episodes: bool = True,
    ):
        super().__init__(buffer_size=buffer_size,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)
        self.chunk_size = chunk_size
        self._valid_beginnings = np.zeros((buffer_size,), dtype=np.bool)
        self._current_episode_length = 0
        self.chunks_can_cross_episodes = chunks_can_cross_episodes

        self._id_range = np.arange(0, self.buffer_size)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self._valid_beginnings[self.pos] = False
        self._current_episode_length += 1
        if self._current_episode_length >= self.chunk_size \
                or self.chunks_can_cross_episodes and len(self) - 1 >= self.chunk_size:
            self._valid_beginnings[(self.pos - self.chunk_size + 1) % self.buffer_size] = True
            # TODO: if we optimize memory usage we don't want to sample transitions FROM terminal state as they are invalid
            #       i.e. if we observe (s_t, a_t, r_t, done=1, s_t+1) then s_t+1 is a terminal state so we should not
            #       return a transition (s_t+1, a_t+1, r_t+1, done=0, s_k) because s_k starts a new transition and
            #       a_t+1, r_t+1 are not even defined.
            #       Question is: is this already taken care of or note? In the non-optimized version it is.
        if done:
            self._current_episode_length = 0

        super().add(obs, next_obs, action, reward, done, infos)

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage or not self.full:
            lower_bound = 0
            upper_bound = self.buffer_size if self.full else self.pos
            return self._sample_from(np.arange(lower_bound, upper_bound), batch_size, env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        return self._sample_from((np.arange(1, self.buffer_size) + self.pos) % self.buffer_size, batch_size, env)

    def _sample_from(self, arange, batch_size, env):
        masked_idx = self._id_range[arange]
        masked_valid = self._valid_beginnings[arange]
        valid_idxs = masked_idx[masked_valid]
        chunk_beginnings = valid_idxs[np.random.randint(0, len(valid_idxs), size=(batch_size,))]
        return self._get_samples(chunk_beginnings, env)

    def _get_samples(self, beginnings: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        chunk_idxs = self._get_chunk_batches(beginnings)
        batch_idxs = chunk_idxs.transpose().reshape(-1)  # Unroll indices
        reshape = Reshaper(len(beginnings), self.chunk_size)
        # return observations.reshape(chunk_size, batch_size, *observations.shape[1:]), self.actions[vec_idxs].reshape(
        #     chunk_size, batch_size, -1), self.rewards[vec_idxs].reshape(chunk_size, batch_size), self.nonterminals[
        #            vec_idxs].reshape(chunk_size, batch_size, 1)

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_idxs + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_idxs, 0, :], env)

        data = (
            reshape(self._normalize_obs(self.observations[batch_idxs, 0, :], env)),
            reshape(self.actions[batch_idxs, 0, :]),
            reshape(next_obs),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            reshape(self.dones[batch_idxs] * (1 - self.timeouts[batch_idxs])),
            reshape(self._normalize_reward(self.rewards[batch_idxs], env)),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_chunk_batches(self, beginnings):
        return np.array([np.arange(idx, idx + self.chunk_size) % self.buffer_size for idx in beginnings])


class Reshaper:
    def __init__(self, batch_size, chunk_size):
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def __call__(self, array):
        return self.reshape(array)

    def reshape(self, array):
        return np.reshape(array, (self.chunk_size, self.batch_size, *array.shape[1:]))
