from typing import Final, Iterable, Optional, Sequence, Tuple
from nptyping import NDArray, Shape, Int, Bool
import numpy as np

Point = Tuple[int, int]

# TODO is this rows, columns or columns, rows
BitMaze = NDArray[Shape["Height, Width"], Int]
"""Maze of 0s and 1s, where 1s are walls.
"""


def add_2d(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def average_2d(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return (a[0] + b[0] // 2, a[1] + b[1] // 2)


def distance_mask(maze: BitMaze, point: Point, distance: int, mask: int = 1) -> BitMaze:
    """Return a new maze with the coordinates around point set to the given mask (1 for wall by default)"""
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


def empty_indices(maze: BitMaze) -> Sequence[Point]:
    return filled_indices(1 - maze)


def filled_indices(maze: BitMaze) -> Sequence[Point]:
    indices = np.transpose(np.nonzero(maze))
    return [(x, y) for y, x in indices]


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


example_maze: Final[BitMaze] = np.array(
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
