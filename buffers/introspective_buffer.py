import numpy as np
from typing import Union, Optional, List, Dict, Any
from buffers.chunk_buffer import ChunkReplayBuffer


class IntrospectiveChunkReplayBuffer(ChunkReplayBuffer):
    def __init__(self, buffer_size: int, *args, **kwargs):
        super().__init__(buffer_size, *args, **kwargs)
        
        self.sample_counts = np.zeros((buffer_size,), dtype=np.int)
        self.first_access = np.zeros((buffer_size,), dtype=np.int) - 1

    def _log_indices(self, indices):
        self.sample_counts[indices] += 1
        mask = np.zeros_like(self.first_access, dtype=bool)
        mask[indices] = 1
        self.first_access[(self.first_access == -1) & mask] = self.pos

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]]
            ):
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_chunk_batches(self, beginnings):
        sampled_indices = super()._get_chunk_batches(beginnings)
        self._log_indices(sampled_indices.flatten())
        return sampled_indices
