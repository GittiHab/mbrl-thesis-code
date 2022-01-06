import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import abc
from utils import figure_to_PIL

MULTIPLE_DIMS = 4
AGENT_DIR_TO_STR = {
    0: '>',
    1: 'v',
    2: '<',
    3: '^'
}  # source: minigrid.MiniGridEnv class in the gym_minigrid package
AGENT_DIR_TO_STR_NORMED = {
    1: '>',
    2: 'v',
    3: '<',
    0: '^'
}


class BaseGridRenderer(abc.ABC):

    def __init__(self, width, height):
        self._width = width
        self._height = height

        self._walls = np.zeros((width, height), dtype=np.uint8)
        self._rewards = np.zeros((width, height), dtype=np.uint8)
        self._start = None

        self.colors = {'walls': (160, 160, 160),
                       'cells': (255, 255, 255),
                       'rewards': (75, 150, 80),
                       'grid': (160, 160, 160),
                       'start': (255, 235, 133),  # yellow
                       # 'start': (224, 25, 25),  # red
                       'visited': (93, 246, 244),
                       'visited_negative': (255, 93, 68)}  # rgb values for different cell types
        self.cell_size = 44  # pixel width of a single cell

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @abc.abstractmethod
    def render(self):
        raise NotImplementedError()

    def set_minigrid_attributes(self, grid):
        from gym_minigrid.minigrid import Goal, Wall
        mapping = {Goal: self._rewards, Wall: self._walls}

        for x in range(grid.width):
            for y in range(grid.height):
                cell_type = type(grid.get(x, y))
                if cell_type in mapping:
                    mapping[cell_type][x, y] = 1

        self._clear_cache()

    def set_start(self, x, y, orientation=None):
        self._start = (x, y) if orientation is None else (x, y, orientation)

    @property
    def start_cell(self):
        return self._start

    def value_at(self, x, y, corner):
        raise NotImplementedError()

    def not_empty(self, cell):
        return self._rewards[cell] or self._walls[cell]

    def is_wall(self, cell, y=None):
        if y is None:
            x, y = cell
        else:
            x = cell
        return self._walls[x, y]

    def _clear_cache(self):
        pass


class GridRenderer(BaseGridRenderer):

    def __init__(self, width, height, multiple=False, use_cache=False):
        super().__init__(width, height)

        self.multiple = multiple
        dims = (width, height, MULTIPLE_DIMS) if multiple else (width, height)
        self._counts = np.zeros(dims, dtype=self.count_dtype)
        self._start = None

        self.channels = 3
        self.center_size = 20
        self.corner_size = int(self.cell_size / 2)
        self.grid_size = 2

        self._cache = None
        self._use_cache = use_cache

    @property
    def count_dtype(self):
        return np.uint8

    def set_start(self, pos):
        self._clear_cache()
        self._start = pos

    def set_visit_counts(self, cell_counts):
        self._clear_cache()
        for cell, counts in cell_counts.items():
            self._counts[cell] = counts

    def value_at(self, x, y, corner):
        pass

    @property
    def pixel_width(self):
        return self._width * self.pixel_cell_size + self.grid_size

    @property
    def pixel_height(self):
        return self._height * self.pixel_cell_size + self.grid_size

    @property
    def pixel_cell_size(self):
        return self.cell_size + self.grid_size

    def pixel_x_y(self, cell, y=None):
        if y is None:
            x, y = cell
        else:
            x = cell
        x = x * self.pixel_cell_size + self.grid_size
        y = y * self.pixel_cell_size + self.grid_size
        return x, y

    def _init_image(self):
        image = np.zeros((self.pixel_width, self.pixel_height, 3), dtype=np.uint8)
        image[:, :] = self.colors['cells']
        return image

    def _draw_grid(self, image):
        # Left and top
        image[:self.grid_size, :] = self.colors['grid']
        image[:, :self.grid_size] = self.colors['grid']
        # Right and bottom of each cell
        for i in np.arange(self.pixel_cell_size, self.pixel_width, self.pixel_cell_size):
            image[i:i + self.grid_size, :] = self.colors['grid']
            image[:, i:i + self.grid_size] = self.colors['grid']
        return image

    def _draw_rewards(self, image):
        self._fill_cells(image, self._rewards, 'rewards')
        return image

    def _draw_visited(self, image):
        max = np.max(self._counts[self._walls == 0])
        min = np.min(self._counts[self._walls == 0])
        if max == 0 and min == 0:
            return image

        for x in np.arange(self._counts.shape[0]):
            for y in np.arange(self._counts.shape[1]):
                if len(self._counts.shape) == 2 and self._counts[x, y] == 0 or np.all(self._counts[x, y] == 0):
                    continue
                x_pixel, y_pixel = self.pixel_x_y(x, y)

                self._fill_visited_cell(image, x_pixel, y_pixel, self._counts[x, y], min, max)
                if self._rewards[x, y]:
                    self._fill_center(image, x_pixel, y_pixel, 'rewards')
        return image

    def _fill_visited_cell(self, image, x, y, count, min, max):
        if not self.multiple:
            return self._fill_cell(image, x, y, self._visited_color(count, min, max))
        for i in range(MULTIPLE_DIMS):
            if count[i] == 0:
                continue
            self._fill_corner(image, x, y, i, self._visited_color(count[i], min, max))
        return image

    def _visited_color(self, count, min, max):
        visited_color = np.array(self.colors['visited']) / max
        if min < 0:
            visited_neg_color = np.array(self.colors['visited_negative']) / np.abs(min)
        return visited_color * (1 + max - count) if count >= 0 else visited_neg_color * (1 + np.abs(min - count))

    def _fill_corner(self, image, x, y, corner, color='rewards'):
        """
        @param corner: From left to right, top to bottom: (0: top left, 1: top right, 2: bottom right, 3: bottom left)
        """
        assert 0 <= corner <= 3, 'Corner must be int in interval [0, 3].'

        left = corner == 0 or corner == 3
        top = corner < 2

        start_x = x if left else x + self.cell_size - self.corner_size
        start_y = y if top else y + self.cell_size - self.corner_size

        end_x = x + self.corner_size if left else x + self.cell_size
        end_y = y + self.corner_size if top else y + self.cell_size

        image[start_x:end_x, start_y:end_y] = self.colors[color] if type(color) is str else color
        return image

    def _fill_center(self, image, x, y, color='rewards'):
        start = int(self.cell_size * 0.5 - self.center_size * 0.5)
        end = int(self.cell_size * 0.5 + self.center_size * 0.5)
        image[x + start:x + end, y + start:y + end] = self.colors[color]
        return image

    def _draw_walls(self, image):
        self._fill_cells(image, self._walls, 'walls')
        return image

    def _fill_cells(self, image, mask, color):
        if np.sum(mask) == 0:
            return image
        for cell, is_wall in np.ndenumerate(mask):
            if not is_wall:
                continue
            x, y = self.pixel_x_y(cell)
            self._fill_cell(image, x, y, color)
        return image

    def _fill_cell(self, image, x, y, color):
        image[x:x + self.cell_size, y:y + self.cell_size] = self.colors[color] if type(color) is str else color
        return image

    def _color_contours(self, image, cell, color):
        x, y = self.pixel_x_y(cell)
        image[x - self.grid_size:x, y - self.grid_size:y + self.pixel_cell_size] = self.colors[color]
        image[x + self.cell_size:x + self.cell_size + self.grid_size, y - self.grid_size:y + self.pixel_cell_size] = \
            self.colors[color]
        image[x - self.grid_size:x + self.pixel_cell_size, y - self.grid_size:y] = self.colors[color]
        image[x - self.grid_size:x + self.pixel_cell_size, y + self.cell_size:y + self.cell_size + self.grid_size] = \
            self.colors[color]
        return image

    def _highlight_start(self, image):
        if self._start is None:
            return image
        image = self._color_contours(image, self._start, 'start')
        return image

    def _set_cache(self, image):
        if self._use_cache:
            self._cache = image

    def _get_cache(self):
        return self._cache

    def _clear_cache(self):
        self._cache = None

    def _to_image_object(self, image):
        return Image.fromarray(image, mode='RGB' if self.channels == 3 else 'RGBA')

    def render(self):
        if self._get_cache() is not None:
            return self._to_image_object(self._get_cache())

        image = self._init_image()
        self._draw_rewards(image)
        self._draw_visited(image)
        self._draw_walls(image)
        self._draw_grid(image)
        self._highlight_start(image)
        self._set_cache(image)
        return self._to_image_object(image.transpose((1, 0, 2)))


class ArbitraryGridRenderer(GridRenderer):

    @property
    def count_dtype(self):
        return np.float

    def _draw_visited(self, image):
        masked = self._counts[self._walls == 0]
        self._min_cache = np.min(masked)
        if self._min_cache < 0:
            self._max_neg_cache = self._min_cache
            self._min_neg_cache = np.max(masked[masked < 0])
            self._min_cache = np.min(masked[masked > 0])
        self._max_cache = np.max(masked)

        return super()._draw_visited(image)

    def _visited_color(self, count, min, max):
        calc_color = lambda color, count, min, max: np.array(color) * (0.4 + 0.6 * (1 - (count - min) / (max - min)))

        if count < 0:
            return calc_color(self.colors['visited_negative'], count, self._min_neg_cache, self._max_neg_cache)
        return calc_color(self.colors['visited'], count, self._min_cache, self._max_cache)


class TrajectoryGridRenderer(BaseGridRenderer):

    def __init__(self, width, height, normed=True, seed=None, size=6, min_alpha=0.25):
        super().__init__(width, height)
        self._trajectories = []
        self._normed = normed
        self._random = np.random.default_rng(seed)
        self.size = size
        self.min_alpha = min_alpha

    def add_trajectory(self, trajectory):
        if len(trajectory.shape) > 2:
            self._trajectories.append([(*self._map_coordinate(*state[:2]), state[2]) for state in trajectory])
        else:
            self._trajectories.append([self._map_coordinate(*state[:2]) for state in trajectory])

    def add_trajectories(self, trajectories):
        for t in trajectories:
            self.add_trajectory(t)

    def _map_coordinate(self, x, y):
        x_new = x
        y_new = self._height - y - 1
        return x_new, y_new

    def _get_color(self, color):
        return np.array(self.colors[color]) / 255

    def positions(self, i=None, margin=0., mask=False, offset_factor=0.):
        if self._normed:
            positions = np.array([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=np.float)
        else:
            positions = np.array([[1, 1, 0, 0], [1, 0, 0, 1]], dtype=np.float)
        if not mask:
            positions[positions == 1] = positions[positions == 1] - 0.25 + margin
            positions[positions == 0] = positions[positions == 0] + 0.25 - margin
            if margin and offset_factor > 0:
                offset_mask = positions[0, :] != positions[1, :]
                positions[1, offset_mask] += margin * (-1 + 2 * positions[0, offset_mask]) * offset_factor
        if i is None:
            return positions[0], positions[1]
        return positions[i]

    def _make_grid(self, axis):
        # based on https://stackoverflow.com/questions/31620774/in-a-scatter-plot-make-each-grid-cell-as-a-square-in-matplotlib-in-python-3-2

        axis.set_xlim(0, self._width)
        axis.set_ylim(0, self._height)

        n_x, n_y = self._width, self._height
        axis.set_aspect(axis.get_xlim()[1] / axis.get_ylim()[1] * n_y / n_x)
        axis.set_xticks(np.linspace(*axis.get_xlim(), num=n_x + 1))
        axis.set_yticks(np.linspace(*axis.get_ylim(), num=n_y + 1))
        axis.yaxis.set_ticklabels([])
        axis.xaxis.set_ticklabels([])
        axis.grid(True, color=self._get_color('grid'))

    def _add_walls(self, axis):
        for x in range(self._width):
            for y in range(self._height):
                if self._walls[x, y]:
                    wall = Rectangle(self._map_coordinate(x, y), 1, 1, color=self._get_color('walls'))
                    axis.add_patch(wall)

    def _add_reward(self, axis):
        for x in range(self._width):
            for y in range(self._height):
                if self._rewards[x, y]:
                    x_mapped, y_mapped = self._map_coordinate(x, y)
                    reward = Circle((x_mapped + 0.5, y_mapped + 0.5), radius=0.25, color=self._get_color('rewards'))
                    axis.add_patch(reward)

    def _add_start(self, axis):
        if self._start is None:
            return
        if len(self._start) == 3:
            x, y, dir = self._start
        else:
            x, y = self._start
            dir = None
        x, y = self._map_coordinate(x, y)
        offset = self.positions(margin=-0.1)[dir] if dir is not None else (0.5, 0.5)
        start = Rectangle((x + offset[0] - 0.2, y + offset[1] - 0.2), 0.4, 0.4, color=self._get_color('start'))
        axis.add_patch(start)

    def _add_trajectories(self, axis):
        colors = plt.cm.rainbow(
            np.linspace(0, 1, len(self._trajectories)))  # based on https://stackoverflow.com/a/38219022
        alphas = np.linspace(self.min_alpha, 1, len(self._trajectories))
        for i, trajectory in enumerate(self._trajectories):
            xs = np.array([s[0] for s in trajectory]) + 0.25 + self._random.random(len(trajectory)) * 0.5
            ys = np.array([s[1] for s in trajectory]) + 0.25 + self._random.random(len(trajectory)) * 0.5

            color = colors[i]  # self._get_color('visited') * (0.2 + self._random.random() * 0.8)

            axis.plot(xs, ys, c=color, alpha=alphas[i])
            if len(xs.shape) == 2:
                direction = np.array([s[2] for s in trajectory])
                for j in range(4):
                    axis.scatter(xs[direction == j], ys[direction == j],
                                 c=[color], marker=self.direction_marker(j), cmap='rainbow', alpha=alphas[i])
            else:
                axis.scatter(xs, ys, c=[color], marker='o', cmap='rainbow', alpha=alphas[i])

    def render(self, return_figure=True, only_env=False):
        fig = plt.figure()
        fig.set_size_inches(self.size, self.size)
        axis = fig.gca()
        plt.rcParams['figure.figsize'] = self.size, self.size

        self._make_grid(axis)
        self._add_walls(axis)
        self._add_reward(axis)
        self._add_start(axis)
        if not only_env:
            self._add_trajectories(axis)
        if return_figure:
            return fig  # e.g use fig.show()
        img = figure_to_PIL(fig)
        plt.close(fig)
        return img

    def direction_marker(self, direction):
        return AGENT_DIR_TO_STR_NORMED[direction] if self._normed else AGENT_DIR_TO_STR[direction]


class ValueGridRenderer(TrajectoryGridRenderer):
    MARKER_ARROW = 'arrows'
    MARKER_STATIC = 'static'
    MARKER_TEXT = 'text'

    def __init__(self, width, height,
                 marker=MARKER_ARROW, normalize=False, normed=False, value_range=None, margin=None,
                 precision=3, size=6, offset=1.6, scale=None, ignore_walls=True, directions=True):
        super().__init__(width, height, normed=normed, size=size)
        dims = (width, height, MULTIPLE_DIMS) if directions else (width, height)
        self._values = np.zeros(dims, dtype=np.float)
        self._directions = directions
        self._markers = marker
        self._ignore_walls = ignore_walls
        if scale is None:
            self.size_scale = 10 if marker != self.MARKER_TEXT else 1.5
        else:
            self.size_scale = scale
        self.offset = offset
        self._normalize = normalize
        if margin is None:
            self._margin = 0 if marker != self.MARKER_TEXT else 0.1
        else:
            self._margin = margin
        self._number_precision = precision
        assert value_range is None or len(value_range) == 2, 'Value range must be None or contain two values.'
        self._value_range = value_range
        if normalize:
            self._values_raw = np.zeros(dims, dtype=np.float)
        else:
            self._values_raw = self._values

    def add_trajectory(self, trajectory):
        raise NotImplementedError()

    def set_values(self, values):
        if type(values) is dict:
            self.set_values_from_dict(values)
        elif isinstance(values, np.ndarray):
            self.set_values_from_array(values)
        else:
            raise Exception('values must be numpy array or dict object.')
        self.normalize()

    def set_values_from_array(self, values):
        assert len(values.shape) >= 2
        for x in range(values.shape[0]):
            for y in range(values.shape[1]):
                self._values_raw[x, y] = values[x, y]

    def set_values_from_dict(self, values):
        for cell, vals in values.items():
            self._values_raw[cell] = vals

    def normalize(self):
        if self._normalize:
            min = np.min(self._values_raw[self._walls == 0])
            max = np.max(self._values_raw[self._walls == 0])
            if min == max:
                self._values = self._values_raw / max if max > 0 else np.zeros_like(self._values_raw)
            else:
                self._values = (self._values_raw - min) / (max - min)

    def values_at(self, x, y):
        if self._directions:
            return self._values[x, y]
        else:
            return np.array([self._values[x, y]])

    def value_at(self, x, y, corner):
        if self._normed:
            corner_index = {'top_left': 0, 'top_right': 1, 'bottom_right': 2, 'bottom_left': 3}
        else:
            corner_index = {'top_left': 3, 'top_right': 0, 'bottom_right': 1, 'bottom_left': 2}
        return self.values_at(x, y)[corner_index[corner]]

    def _add_text(self, axis, x, y, direction, value, color):
        alignment = ['left', 'right', 'right', 'left'] if self._normed else ['right', 'right', 'left', 'left']
        axis.text(x, y, np.round(value, self._number_precision),
                  horizontalalignment=alignment[direction] if self._directions else 'center',
                  verticalalignment='center',
                  fontsize=10 * self.size_scale, color=color)

    def _label_locations(self, dim, pos):
        if self._directions:
            return self.positions(dim, self._margin, offset_factor=self.offset) + np.array([pos] * MULTIPLE_DIMS)
        return np.array([0.5 + pos])

    def _add_trajectories(self, axis):
        if self._value_range is None:
            min = np.min(self._values[self._walls == 0])
            max = np.max(self._values[self._walls == 0])
        else:
            min, max = self._value_range
        for x in range(self._values.shape[0]):
            for y in range(self._values.shape[1]):
                values = self.values_at(x, y)
                if np.all(values == 0) or self._ignore_walls and self.is_wall(x, y):
                    continue
                x_mapped, y_mapped = self._map_coordinate(x, y)
                xs = self._label_locations(0, x_mapped)
                ys = self._label_locations(1, y_mapped)
                colors = plt.cm.rainbow((values - min) / (max - min) if min != max else values * 0. + 1.)
                if self._markers == ValueGridRenderer.MARKER_STATIC:
                    axis.scatter(xs, ys, c=colors, s=values * self.size_scale, marker='o')
                elif self._markers == ValueGridRenderer.MARKER_ARROW:
                    for dir in range(len(values)):
                        axis.scatter([xs[dir]], [ys[dir]],
                                     c=[colors[dir]], s=[values[dir] * self.size_scale],
                                     marker=self.direction_marker(dir))
                else:
                    for dir in range(len(values)):
                        self._add_text(axis, xs[dir], ys[dir], dir, values[dir], colors[dir])
