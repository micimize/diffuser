from enum import Enum
from optparse import Option
from typing import Callable, Literal, Sequence, Set, cast
from typing import Iterable, Union
from typing import Final, Optional, Tuple, TypedDict
from horizon_diffuser.maze_env.bit_maze import BitMaze, distance_mask, empty_indices, filled_indices
from horizon_diffuser.maze_env.generators import BitMazeGenerator
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, Goal, Wall
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS

from nptyping import NDArray, Shape, Int, Bool
import numpy as np

Point = Tuple[int, int]


class DynamicMaxSteps(Enum):
    FULL_TRAVERSAL_HEURISTIC = "FULL_TRAVERSAL_HEURISTIC"
    NUMBER_OF_EMPTY_SPACES = "NUMBER_OF_EMPTY_SPACES"


# :( why
AGENT_OBJECT_MARKER: Final[None] = None

# TODO Especially for larger mazes in an obscured world, the reward mechanism may get "unfair."
#  I.e. too much luck involved in a good reward.
#  There are a few options for circumventing this if need be:
#  *  Hint: Add indicator of absolute goal position, relative proximity, or direction of
#  *  Jump mechanic: action to see over walls Nx vision away in your current direction.
#  *  Beacon:
#  To circumvent this, a
#
#  Additionally, normalizing the loss against an algorithmically optimal solver would be nice.
#  Not sure how fair such a mechanism would need to be to be useful. I.e. Diyjksjtra's has an advantage.
class MiniGridMazeEnv(MiniGridEnv):
    def __init__(
        self,
        maze_generator: BitMazeGenerator,
        # kwargs
        agent_view_size: int = 5,
        minimum_goal_distance: Optional[int] = None,
        max_steps: Union[int, DynamicMaxSteps] = DynamicMaxSteps.FULL_TRAVERSAL_HEURISTIC,
        see_through_walls: bool = False,
        render_mode: Optional[str] = None,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        self.maze_generator = maze_generator
        self.current_bit_maze: Optional[BitMaze] = None

        self._dynamic_max_steps = None
        if isinstance(max_steps, DynamicMaxSteps):
            self._dynamic_max_steps = max_steps
            max_steps = -1

        # TODO dynamic sizing?
        width = maze_generator.width
        height = maze_generator.height
        if minimum_goal_distance is not None:
            assert minimum_goal_distance <= int(min([width, height]) / 2), (
                f"{minimum_goal_distance} must be lower than half the lowest dimension ({int(min([width, height]))}), "
                f"or else it will be impossible to satisfy when one point is at the center of the maze"
            )

        self.minimum_goal_distance = minimum_goal_distance

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
    def _gen_mission():
        return "traverse the maze to get to the goal"

    @property
    def width(self) -> int:
        if self.current_bit_maze is not None:
            return self.current_bit_maze.shape[0]
        return self.maze_generator.width

    @property
    def height(self) -> int:
        if self.current_bit_maze is not None:
            return self.current_bit_maze.shape[1]
        return self.maze_generator.height

    @property
    def number_of_empty_spaces(self) -> int:
        if self.current_bit_maze is None:
            return 0

        return np.count_nonzero(1 - self.current_bit_maze)

    def steps_to_traverse_every_empty_space(self, backtracking_multiplier: Union[int, float] = 2) -> int:
        return int(backtracking_multiplier * self.number_of_empty_spaces)

    def _set_max_steps(self, max_steps: Optional[int] = None):
        if max_steps is not None:
            self._dynamic_max_steps = None
            self.max_steps = max_steps
            return

        if self._dynamic_max_steps is None:
            return

        if self._dynamic_max_steps == DynamicMaxSteps.FULL_TRAVERSAL_HEURISTIC:
            self.max_steps = self.steps_to_traverse_every_empty_space()
        elif max_steps == DynamicMaxSteps.NUMBER_OF_EMPTY_SPACES:
            self.max_steps = self.number_of_empty_spaces

    def _gen_grid(self, width: int, height: int):
        maze = self.current_bit_maze = self.maze_generator.generate()
        width, height = maze.shape
        self._set_max_steps()

        self.grid = Grid(width, height)
        # Walls
        wall_indices = filled_indices(self.current_bit_maze)
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

    def _get_empty_point(
        self,
        excluding_around: Optional[Tuple[Point, int]] = None,
        excluding_points: Iterable[Point] = tuple(),
    ) -> Point:
        maze = self.current_bit_maze
        if excluding_around:
            point, distance = excluding_around
            maze = distance_mask(maze, point, distance)
        empty = [e for e in empty_indices(maze) if e not in excluding_points]
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
            ), f"Impossible to obtain {min_distance} in {self.current_bit_maze}"
            one = self._get_empty_point(excluding_points=exclude)
            two = self._get_empty_point(excluding_around=(one, min_distance))
            if two != (-1, -1):
                return (one, two)
            exclude.add(one)
