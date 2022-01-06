from argparse import ArgumentParser

import os.path

import torch
from algos.PlaNet.planet import PlaNet
from main import build_env
from visualization.values import *


def visualize(path, name, scale, device, args, bfs=False, values=True, format='png'):
    env, _ = build_env(args)
    env.reset()  # reset environment to get layout of current seed.
    model = PlaNet.load(path, env=env, device=device)

    renderer_cls = ValueRenderer if not bfs else BFSValueRenderer
    renderer = renderer_cls(model.policy.reward_space.values if values else model.policy.reward_space.rewards,
                                env, model.policy.world_model, action_input=not values)
    grid = renderer.render(scale=scale)
    grid.render().savefig(os.path.join(os.path.dirname(path), '{}.{}'.format(name, format)), format=format)


def setup_parser(parser):
    parser.add_argument('--path', '-p', type=str, help='Path to model to be loaded.')
    parser.add_argument('--name', '-n', type=str, default='values', help='Filename of the output.')
    parser.add_argument('--env', '-e', type=str, help='Environment name that should rendered.')
    parser.add_argument('--env_seed', '-s', type=int, help='Seed of the environment.')
    parser.add_argument('--scale', '-S', type=float, default=1.5, help='Scale of labels.')
    parser.add_argument('--rewards', '-r', action='store_true', help='Plot rewards instead of values.')
    parser.add_argument('--bfs', '-b', action='store_true', help='Calculate belief using BFS instead of empty belief.')
    parser.add_argument('--format', '-f', type=str, choices=['png', 'svg', 'pdf', 'jpg'], default='png',
                        help='Format of the rendered figure (Default: png).')


if __name__ == '__main__':
    parser = ArgumentParser(description='Plotting')
    setup_parser(parser)
    args = parser.parse_args()

    if args.path is None:
        args.path = input('Path: ')
        if input('Change name? (yes/no): ').lower() in ('yes', 'y'):
            args.name = input('Name: ')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    visualize(args.path, args.name, args.scale, device, args, bfs=args.bfs, values=not args.rewards, format=args.format)
