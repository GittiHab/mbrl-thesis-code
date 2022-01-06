import torch
from queue import Queue

from torch.nn.functional import one_hot
from algos.PlaNet.world_model import DreamerModel
from type_aliases import Cell
from visualization.grid import ValueGridRenderer
from debug import *

__all__ = ['ValueRenderer', 'BFSValueRenderer']


class ValueRenderer(EnvDebug):
    def __init__(self, model, env, world_model: DreamerModel, action_input=False):
        super().__init__(world_model, env)
        self.model = model
        self.action_input = action_input

    def render(self, return_raw=False, scale=2):
        grid = ValueGridRenderer(self._env.grid.width, self._env.grid.height,
                                 normalize=False, directions=False, **{'marker': 'text', 'size': 20, 'scale': scale})
        grid.set_minigrid_attributes(self._env.grid)
        grid.set_start(*self._env.agent_pos)
        values = {}
        for transition in self._iterate_cells(grid):
            # for x, y, belief, state, action in self._iterate_cells(grid):
            prediction = self._predict(*transition.merged_states, transition.action.unsqueeze(0))
            if len(prediction) > 1:
                prediction = prediction[0]
            values[transition.next_observation] = prediction.item()
        grid.set_values(values)
        if return_raw:
            return grid, return_raw
        return grid

    def _iterate_cells(self, grid):
        for x in range(self._env.grid.width):
            for y in range(self._env.grid.height):
                if grid.is_wall(x, y):
                    continue
                belief, posterior_state = self._retrieve(x, y, self._initial_state())
                yield Transition(self._empty_state(),
                                 self._empty_tensor(self.action_size, self.device),
                                 (x, y),
                                 State(belief, posterior_state))

    def _predict(self, belief, state, action):
        if self.action_input:
            return self.model(belief, state, action)
        return self.model(belief, state)


class BFSValueRenderer(ValueRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _action(self, a):
        return one_hot(torch.tensor([a], device=self.device), 4)

    def _iterate_cells(self, grid):
        queue = Queue()
        queue.put((grid.start_cell, self._initial_state()))
        visited = [grid.start_cell]
        while not queue.empty():
            cell_current, state_current = queue.get()
            belief, posterior_state = self._retrieve(*cell_current, state_current)
            for x_diff, y_diff, action in [(1, 0, 1), (0, 1, 2), (0, -1, 3), (-1, 0, 0)]:
                neighbor = Cell(cell_current[0] + x_diff, cell_current[1] + y_diff)
                if grid.is_wall(neighbor.x, neighbor.y) or neighbor in visited:
                    continue
                visited.append(neighbor)
                queue.put((neighbor, UninferredState(belief, posterior_state, self._action(action), self.device)))

            yield Transition(State(state_current.belief, state_current.state),
                             state_current.action,
                             (cell_current[0], cell_current[1]),
                             State(belief, posterior_state))
            # yield cell_current[0], cell_current[1], belief, posterior_state, state_current.action
