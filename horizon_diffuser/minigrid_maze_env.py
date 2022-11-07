from optparse import Option
from typing import Sequence, Set, cast
from typing import Iterable, Union
from typing import Final, Optional, Tuple, TypedDict
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, Goal, Wall
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS

from nptyping import NDArray, Shape, Int, Bool
import numpy as np

# TODO is this rows, columns or columns, rows
BitMaze = NDArray[Shape["Height, Width"], Int]

Point = Tuple[int, int]

# :( why
AGENT_OBJECT_MARKER: Final[None] = None


class MiniGridMazeEnv(MiniGridEnv):
    def __init__(
        self,
        bit_maze: BitMaze,
        # kwargs
        agent_view_size: int = 5,
        minimum_goal_distance: Optional[int] = None,
        max_steps: Optional[int] = None,
        see_through_walls: bool = False,
        render_mode: Optional[str] = None,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        bit_maze = bordered(bit_maze)

        # TODO random world generation should be done by the env itself based on seed
        height = len(bit_maze)
        width = len(bit_maze[0])
        self.bit_maze = bit_maze
        self.number_of_empty_spaces = np.count_nonzero(1 - self.bit_maze)
        if minimum_goal_distance is not None:
            assert minimum_goal_distance <= int(min([width, height]) / 2), (
                f"{minimum_goal_distance} must be lower than half the lowest dimension ({int(min([width, height]))}), "
                f"or else it will be impossible to satisfy when one point is at the center of the maze"
            )

        self.minimum_goal_distance = minimum_goal_distance

        if max_steps is None:
            max_steps = self.steps_to_traverse_every_empty_space()

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # kwargs
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,
            render_mode=render_mode,
            highlight=highlight,
            tile_size=tile_size,
            agent_pov=agent_pov,
        )

    def steps_to_traverse_every_empty_space(self, backtracking_multiplier: Union[int, float] = 2) -> int:
        return int(backtracking_multiplier * self.number_of_empty_spaces)

    @staticmethod
    def _gen_mission():
        return "traverse the maze to get to the goal"

    def _get_empty_point(
        self,
        excluding_around: Optional[Tuple[Point, int]] = None,
        excluding_points: Iterable[Point] = tuple(),
    ) -> Point:
        maze = self.bit_maze
        if excluding_around:
            point, distance = excluding_around
            maze = _distance_mask(maze, point, distance)
        empty = [e for e in _empty_indices(maze) if e not in excluding_points]
        if len(empty) == 0:
            if not excluding_around:
                raise ValueError(f"Invalid maze: No empty place in the bitmaze {maze}")
            return (-1, -1)
        return cast(Point, tuple(self._rand_elem(empty)))

    def _get_two_empty_points(self, min_distance: Optional[int]) -> Tuple[Point, Point]:
        if not min_distance:
            return (self._get_empty_point(), self._get_empty_point())

        exclude: Set[Point] = set()
        while True:
            assert (
                len(exclude) < self.number_of_empty_spaces
            ), f"Impossible to obtain {min_distance} in {self.bit_maze}"
            one = self._get_empty_point(excluding_points=exclude)
            two = self._get_empty_point(excluding_around=(one, min_distance))
            if two != (-1, -1):
                return (one, two)
            exclude.add(one)

    def _gen_grid(self, width: int, height: int):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Walls
        wall_indices = _filled_indices(self.bit_maze)
        print(f"placing {len(wall_indices)} wall blocks")
        for x, y in wall_indices:
            self.put_obj(Wall(), x, y)

        min_distance = self.minimum_goal_distance
        first_point, second_point = self._get_two_empty_points(min_distance)

        is_agent_placed_first = (min_distance is None) or self._rand_bool()
        if is_agent_placed_first:
            self.put_agent(first_point)
            goal_pos = self.put_obj(Goal(), *second_point)
        else:
            goal_pos = self.put_obj(Goal(), *first_point)
            self.put_agent(second_point)

        print(f"goal placed at {goal_pos}, agent placed at {self.agent_pos}")

    def put_agent(self, pos: Point, agent_obj=None, rand_dir: bool = True) -> None:
        # TODO is this even true? I feel like agents are none as a hack so they can be on top of doors
        assert self.grid.get(*pos) is None, "Cannot place agent on top of another object"
        agent_obj = AGENT_OBJECT_MARKER
        self.grid.set(pos[0], pos[1], agent_obj)
        self.agent_pos = pos
        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)


def _distance_mask(maze: BitMaze, point: Point, distance: int, mask: int = 1) -> BitMaze:
    height = len(maze)
    width = len(maze[0])
    masked = maze.copy()
    x, y = point

    x_low = min([x - distance, 0])
    x_high = max([x + distance + 1, width])
    y_low = min([y - distance, 0])
    y_high = max([y + distance + 1, height])
    masked[x_low:x_high, y_low:y_high] = mask
    return masked


def _empty_indices(maze: BitMaze) -> Sequence[Point]:
    return _filled_indices(1 - maze)


def _filled_indices(maze: BitMaze) -> Sequence[Point]:
    indices = np.transpose(np.nonzero(maze))
    return [(x, y) for y, x in indices]


# TODO RNG class for reproducibility
# TODO remove borders
def random_maze(width: int = 81, height: int = 51, complexity: float = 0.75, density: float = 0.75) -> BitMaze:
    r"""Generate a random maze array.

    It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
    is ``1`` and for free space is ``0``.

    Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm

    taken from https://github.com/zuoxingdong/mazelab/blob/master/mazelab/generators/random_maze.py
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z: NDArray[Shape["Height, Width"], Bool] = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = (
            np.random.randint(0, shape[1] // 2 + 1) * 2,
            np.random.randint(0, shape[0] // 2 + 1) * 2,
        )
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_

    return Z.astype(int)


def bordered(maze: BitMaze) -> BitMaze:
    edge_blocks_without_borders = sum(
        [
            np.count_nonzero(1 - maze[0, :]),
            np.count_nonzero(1 - maze[-1, :]),
            np.count_nonzero(1 - maze[:, 0]),
            np.count_nonzero(1 - maze[:, -1]),
        ]
    )
    if edge_blocks_without_borders == 0:
        return maze
    return np.pad(maze, pad_width=1, mode="constant", constant_values=1)


example_maze: Final = np.array(
    [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    ]
)
