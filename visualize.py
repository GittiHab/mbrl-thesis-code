from argparse import ArgumentParser

import os.path

import numpy as np

from main import build_env
from visualization.trajectories import ExperienceRenderer
from stable_baselines3.dqn.policies import DQNPolicy
from buffers.utils import find_episode_ending
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


def visualize(args, start, end, format='png'):
    env, _ = build_env(args)
    env.reset()  # reset environment to get layout of current seed.
    model = OffPolicyAlgorithm('MlpPolicy', env, DQNPolicy, 1e-5)  # last two parameters are just placeholders!
    model.load_replay_buffer(args.path)

    replay_buffer = model.replay_buffer

    renderer = ExperienceRenderer(replay_buffer, env, args.time_axis)
    if args.timesteps:
        start, end, data = find_episode_ending(replay_buffer, start, end)
        print('Visualizing episodes {} ({}) to {} ({}).'.format(start, data[0], end, data[1]))
    grid, visited_count, legal_count = renderer.render_range(start, end + 1 if end >= 0 else end, args.stats)
    print('Visited:', visited_count, 'out of', legal_count, '({}%)'.format(
        np.round(float(visited_count) / legal_count * 100, 3)))
    if args.stats:
        return
    grid.render().savefig(os.path.join(os.path.dirname(args.path), '{}.{}'.format(args.name, format)), format=format)


def setup_parser(parser):
    parser.add_argument('--path', '-p', type=str, help='Path to replay buffer.')
    parser.add_argument('--name', '-n', type=str, default='trajectories', help='Filename of the output.')
    parser.add_argument('--env', '-e', type=str, help='Environment name that should rendered.')
    parser.add_argument('--env_seed', '-s', type=int, help='Seed of the environment.')

    parser.add_argument('--timesteps', '-ts', action='store_true', help='Index by time steps instead of episodes.')
    parser.add_argument('--range', '-r', type=str,
                        help='Episode range to render. '
                             'Start and end separated by dash ("-") or one number for single episode.')
    parser.add_argument('--time-axis', '-ta', action='store_true',
                        help='Visualize time dimension by fading out older trajectories.')
    parser.add_argument('--format', '-f', type=str, choices=['png', 'svg', 'pdf', 'jpg'], default='png',
                        help='Format of the rendered figure (Default: png).')
    parser.add_argument('--stats', action='store_true', help='Only print stats. No rendering.')

    parser.add_argument('--interactive', '-i', action='store_true',
                        help='In interactive mode input values are asked by the program.')


if __name__ == '__main__':
    parser = ArgumentParser(description='Plotting')
    setup_parser(parser)
    args = parser.parse_args()

    if args.interactive:
        if args.path is None:
            args.path = input('Path: ')
        if args.range is None:
            args.range = input('Range: ')
        if input('Change name? (yes/no): ').lower() in ('yes', 'y'):
            args.name = input('Name: ')

    if os.path.isdir(args.path):
        args.path = os.path.join(args.path, 'replay_buffer.pkl')

    if args.range is not None and args.range != '':
        range = args.range.split('-')
        assert len(range) <= 2, 'Range must be two integers separated by an dash but was {}'.format(args.range)
        if len(range) == 1:
            start = end = int(range[0])
        else:
            start, end = [int(i) for i in range]
            assert start <= end, 'Start ({}) of range must be smaller than end ({}).'.format(start, end)
    else:
        start, end = 0, -1

    visualize(args, start, end, format=args.format)
