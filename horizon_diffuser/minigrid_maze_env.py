from typing import Union
from typing import Final, Optional, Tuple, TypedDict
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, Goal, Wall
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS

from nptyping import NDArray, Shape, Int, Bool
import numpy as np

# TODO is this rows, columns or columns, rows
BitMapMaze = NDArray[Shape["Height, Width"], Int]

Point = Tuple[int, int]

# :( why
AGENT_OBJECT_MARKER: Final[None] = None


class MiniGridMazeEnv(MiniGridEnv):
    """
    grid_size: int = None,
    width: int = None,
    height: int = None,
    max_steps: int = 100,
    see_through_walls: bool = False,
    agent_view_size: int = 7,
    render_mode: Optional[str] = None,
    highlight: bool = True,
    tile_size: int = TILE_PIXELS,
    agent_pov: bool = False,
    """

    def __init__(
        self,
        bit_map: BitMapMaze,
        start_pos: Point,
        goal_pos: Point,
        # kwargs
        agent_view_size: int = 5,
        max_steps: Optional[int] = None,
        see_through_walls: bool = False,
        render_mode: Optional[str] = None,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):

        # TODO random world generation should be done by the env itself based on seed
        wall_width = 2
        height = len(bit_map) + wall_width
        width = len(bit_map[0]) + wall_width
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.bit_map = bit_map

        if max_steps is None:
            max_steps = self.steps_to_traverse_every_empty_space(bit_map)

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

    @staticmethod
    def steps_to_traverse_every_empty_space(maze: BitMapMaze, backtracking_multiplier: Union[int, float] = 2) -> int:
        empty_spaces = np.count_nonzero(1.0 - maze)
        return int(backtracking_multiplier * empty_spaces)

    @staticmethod
    def _gen_mission():
        return "traverse the maze to get to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Goal
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        # Agent
        self.place_agent_at_pos(0, self.start_pos)

        # Walls
        for x in range(self.bit_map.shape[0]):
            for y in range(self.bit_map.shape[1]):
                if self.bit_map[y, x]:
                    # Add an offset of 1 for the outer walls
                    self.put_obj(Wall(), x + 1, y + 1)

    def place_agent_at_pos(self, pos: Point, agent_obj=None, rand_dir: bool = True) -> None:
        # TODO is this even true? I feel like agents are none as a hack so they can be on top of doors
        assert self.grid.get(*pos) is None, "Cannot place agent on top of another object"
        agent_obj = AGENT_OBJECT_MARKER

        self.grid.set(pos[0], pos[1], agent_obj)

        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)


# Todo RNG class for reproducibility
def random_maze(width: int = 81, height: int = 51, complexity: float = 0.75, density: float = 0.75) -> BitMapMaze:
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


example_map: Final = np.array(
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
