from typing import Iterable, List, Optional, Tuple, TypeVar, Union
import numpy as np
from typing import Protocol
from nptyping import NDArray, Shape, Bool
from numpy.random import Generator

from gymnasium.core import seeding

from horizon_diffuser.maze_env.bit_maze import BitMaze, average_2d, add_2d


_Seed = Union[int, List[int]]
# TODO https://github.com/john-science/mazelib is very promising

T = TypeVar("T")


class BitMazeGenerator(Protocol):
    np_random: Generator
    # TODO should be dynamic
    width: int
    height: int

    def __init__(self, seed: _Seed, width: int, height: int):
        self.width = width
        self.height = height
        self.np_random, _seed = seeding.np_random(seed)

    def reset(self, seed: Optional[_Seed] = None) -> None:
        if seed is not None:
            #  TODO _seed here is the entropy, idk if that's the same or if entropy changes or what.
            self.np_random, _seed = seeding.np_random(seed)

    def generate(self) -> BitMaze:
        pass

    def _rand_int(self, low: int, high: int) -> int:
        """Generate random integer in [low,high)"""
        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """Generate random float in [low,high)"""
        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """Generate random boolean value"""
        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """Pick a random element in a list"""
        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> List[T]:
        """Sample a random subset of distinct elements of a list"""
        lst = list(iterable)
        assert num_elems <= len(lst)

        out: List[T] = []
        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _random_point(self, padding: int = 0) -> Tuple[int, int]:
        return (self._rand_int(padding, self.width - padding), self._rand_int(padding, self.height - padding))


class DFSWallIslandGenerator(BitMazeGenerator):
    """Depth-first search / recursive backtracking algorithm from an [ancient wikipedia example].

    Creates a ``number_of_islands`` are created of ``island_size`` wall blocks,
    derived from ``island_density`` and ``island_scale`` respectively.
    With a low ``island_scale``, islands are very small and the maze is easy to solve.
    With low density, the maze has more "big empty rooms".

    Each island is created by:
    1. Choosing a random starting point with odd coordinates
    2. Choosing
    If the cell two steps in the direction is free, then a wall is added at both steps in this direction.
    The process is iterated for ``island_size`` steps for this island.

    [ancient wikipedia example]: https://en.wikipedia.org/w/index.php?title=Maze_generation_algorithm&oldid=926968112#Python_code_example
    """

    def __init__(self, seed: _Seed, height: int, width: int, island_scale: float, island_density: float):
        assert (
            width % 2 == 1 and height % 2 == 1
        ), f"This algorithm is designed for odd dimensions but got ({width}, {height})"
        super().__init__(seed, height, width)
        self.island_scale = island_scale
        self.island_density = island_density

    @property
    def island_size(self):
        scale_factor = 5
        return int(self.island_scale * (scale_factor * (self.width + self.height)))

    @property
    def number_of_islands(self):
        return int(self.island_density * ((self.height // 2) * (self.width // 2)))

    def add_wall_island(self, maze: BitMaze, island_size: int):
        width, height = maze.shape
        # always starting with odd coordinates prevents 2+ thickness walls
        x, y = (self._rand_int(0, width // 2 + 1), self._rand_int(0, height // 2 + 1))
        maze[x, y] = 1

        two_left = (0, -2)
        two_right = (0, 2)
        two_up = (-2, 0)
        two_down = (2, 0)
        for j in range(island_size):
            growth_candidates: List[Tuple[int, int]] = []
            if x > 1:
                growth_candidates.append(two_left)
            if x < width - 2:
                growth_candidates.append(two_right)
            if y > 1:
                growth_candidates.append(two_up)
            if y < height - 2:
                growth_candidates.append(two_down)

            growth_candidates = [c for c in growth_candidates if maze[add_2d((x, y), c)] == 0]
            if len(growth_candidates) > 0:
                dir = self._rand_elem(growth_candidates)
                new_point = add_2d((x, y), dir)
                maze[new_point] = 1
                mid_point = add_2d((x, y), average_2d(dir, (0, 0)))
                maze[mid_point] = 1
                x, y = new_point

    def generate(self):
        maze: NDArray[Shape["Height, Width"], Bool] = np.zeros((self.width, self.height), dtype=bool)
        maze[0, :] = maze[-1, :] = 1
        maze[:, 0] = maze[:, -1] = 1

        size = self.island_size

        for _island in range(self.number_of_islands):
            self.add_wall_island(maze, size)

        return maze
