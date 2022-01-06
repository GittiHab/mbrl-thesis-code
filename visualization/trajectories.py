from visualization.grid import TrajectoryGridRenderer, GridRenderer
from buffers.utils import retrieve_episodes
import numpy as np


class ExperienceRenderer:

    def __init__(self, memory, env, time=False):
        self.memory = memory
        self._env = env
        self.time = time

    def render_range(self, start_episode, end_episode, skip_render=False):
        episodes = retrieve_episodes(self.memory, start_episode, end_episode)
        if not skip_render:
            grid = self._render_trajectories(trajectories=episodes)
        else:
            grid = None
        unique_counts, legal_count = self.visit_stats(trajectories=episodes)
        return grid, unique_counts, legal_count

    def visit_stats(self, trajectories):
        from gym_minigrid.minigrid import Wall
        visitable_states = 0
        for i in range(self._env.grid.width):
            for j in range(self._env.grid.height):
                visitable_states += not isinstance(self._env.grid.get(i, j), Wall)
        all_visited_states = np.concatenate(trajectories)
        unique_elements = set([tuple(all_visited_states[i]) for i in range(all_visited_states.shape[0])])
        return len(unique_elements), visitable_states

    def _render_trajectories(self, trajectories):
        renderer = TrajectoryGridRenderer(self._env.grid.width, self._env.grid.height, normed=False,
                                          min_alpha=0.25 if self.time else 1)
        renderer.set_minigrid_attributes(self._env.grid)
        renderer.add_trajectories(trajectories)
        return renderer  #renderer.render(return_figure=False)

    def _render_counts(self, counts, start):
        renderer = GridRenderer(self._env.grid.width, self._env.grid.height, multiple=True)
        renderer.set_start(start)
        renderer.set_minigrid_attributes(self._env.grid)
        renderer.set_visit_counts(counts)
        return renderer.render()
