import os
from typing import Iterable

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

__all__ = ['Checkpointing', 'CheckpointAfterCallback', 'NullCheckpointing']


class Checkpointing:

    @staticmethod
    def from_config(config):
        return Checkpointing(config.path,
                             config.prefix,
                             after=None if 'at' not in config else config.at,
                             every=None if 'every' not in config else config.every,
                             exploration=config.exploration_phase,
                             exploitation=config.exploitation_phase)

    def __init__(self, path, prefix, after=None, every=None, exploration=True, exploitation=True):
        self.path = path
        self.prefix = prefix
        self.after = after
        self.every = every

        self.exploration = exploration
        self.exploitation = exploitation

        if every is not None:
            self._every_callback = CheckpointCallback(every, save_path=self.path, name_prefix=self.prefix)
        if after is not None:
            self._after_callback = CheckpointAfterCallback(after, save_path=self.path, name_prefix=self.prefix)

    def _update_prefix(self, prefix):
        if self.every is not None:
            self._every_callback.name_prefix = prefix
        if self.after is not None:
            self._after_callback.name_prefix = prefix

    def get_callback(self, suffix=None):
        prefix = self.prefix + '_' + suffix if suffix else self.prefix
        self._update_prefix(prefix)

        if self.after is None:
            return self._every_callback
        if self.every is None:
            return self._after_callback
        return CallbackList([self._every_callback, self._after_callback])

    def get_exploration_callback(self):
        if self.exploration:
            return self.get_callback('explore')
        return None

    def get_exploitation_callback(self):
        if self.exploitation:
            return self.get_callback('exploit')
        return None


class CheckpointAfterCallback(CheckpointCallback):
    """
    Callback for saving a model after ``save_after`` calls to ``env.step()``.
    Preserves the counts across learning iterations, i.e., if .learn() is called twice with this callback the model is
    not saved twice after N steps.

    Based on the original CheckpointCallback class in the SB3.

    NOTE: Model is not saved if a file with the same name already exists.
          This should save time when using in combination with the CheckpointCallback that saves model frequently.

    :param save_after:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_after: Iterable[int], save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(1, save_path, name_prefix, verbose)
        self.save_after = np.array(save_after)

    def _decrement(self):
        self.save_after -= 1

    def _on_step(self) -> bool:
        self._decrement()
        if 0 in self.save_after:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            if os.path.isfile(path):
                if self.verbose > 1:
                    print(f"Skipping saving to {path}. File already exists.")
                return True
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


class NullCheckpointing(Checkpointing):

    def __init__(self):
        super(NullCheckpointing, self).__init__('', '')

    def get_callback(self, suffix=None):
        return None
