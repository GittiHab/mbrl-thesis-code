import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


def find_episode_ending(memory: ReplayBuffer, start=None, end=None):
    """
    Find start and end episodes based on *timesteps*.
    :param start: Time step where to start (first episode *after* this timestep is returned).
    :param end: Time step where to end (first episode *after* this timestep is returned).
    :return: two episode indices.
    """
    start_ep = 0
    end_ep = int(memory.dones.sum() - 1)
    start_i = start
    end_i = end if end != -1 else memory.size()

    dones = memory.dones.squeeze(-1)
    episodes_beginnings = np.arange(len(dones))[dones == 1] + 1
    episodes_beginnings = np.insert(episodes_beginnings, 0, 0)

    if start is not None:
        start_ep = np.argmax(episodes_beginnings >= start)
        start_i = episodes_beginnings[start_ep]
    if end is not None and end != -1:
        end_ep = np.argmax(episodes_beginnings >= end)
        end_i = episodes_beginnings[end_ep]


    return start_ep, end_ep, (start_i, end_i)


def retrieve_episodes(memory: ReplayBuffer, start, end):
    """
    Retrieve the observations in the trajectories of the given episodes.
    Interface is analogous to range(), so start is inclusive and end is exclusive.

    :param memory: The replay buffer from which the episodes should be retrieved.
    :param start: First episode number that should be retrieved.
    :param end: First episode number that should not be retrieved (i.e., it is exclusive).
    :return Observations of episodes [start, end).
    """
    if memory.full and memory.pos > 0:
        raise Exception('Cannot reconstruct as memory has been filled over its size.')

    episode_starts, num_episodes = get_episode_info(memory)
    assert end <= num_episodes, 'There are no {} episodes, only {}. Note, that episodes are 0 indexed.'.format(end, num_episodes)

    trajectories = []
    for i, trajectory_start in enumerate(episode_starts[start:end]):
        trajectory = memory.observations[trajectory_start:episode_starts[start + i + 1], 0]
        last_observation = memory.next_observations[episode_starts[start + i + 1] - 1, 0].reshape(1, -1)
        trajectories.append(np.append(trajectory, last_observation, axis=0))

    return trajectories


def get_episode_info(memory: ReplayBuffer):
    episode_starts = np.arange(0, memory.buffer_size)[memory.dones[:, 0].flatten() == 1] + 1
    num_episodes = len(episode_starts[episode_starts < memory.pos]) + 1
    episode_starts = [0] + list(episode_starts)
    return episode_starts, num_episodes
