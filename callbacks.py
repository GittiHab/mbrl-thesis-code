from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

__all__ = ['LogEpisodeReward']


class LogEpisodeReward(BaseCallback):
    """
    Allows to log the reward and length of every episode.
    This is only sensible when the log is dumped after every episode.
    """

    def __init__(self, logger, verbose=0):
        super().__init__(verbose)
        self.logger = logger

    def _item(self, key):
        if isinstance(self.locals[key], np.ndarray):
            return self.locals[key].item()
        return self.locals[key]

    def _on_step(self):
        if self._item('done'):
            self.logger.record("rollout/reward", self._item('episode_reward') + self._item('reward'))
            self.logger.record("rollout/length", self.locals['episode_timesteps'])

        return True
