# %%
from typing import Final, Iterable, List, Optional, Sequence, Tuple
from minigrid_maze_env import BitMaze, MiniGridMazeEnv, example_maze, random_maze
import PIL
from IPython import display

import numpy as np

image: Final = PIL.Image.fromarray

complexity_space: List[float] = list(np.arange(0.0, 1.1, 0.25))
density_space: List[float] = list(np.arange(0.0, 1.1, 0.25))
width = 20
height = 20


def sample_mazes(complexities: Sequence[float], densitiies: Sequence[float]) -> Iterable[BitMaze]:
    for complexity in complexities:
        for density in densitiies:
            print(f"complexity {complexity} density {density}")
            yield random_maze(50, 50, complexity, density)


env = MiniGridMazeEnv(random_maze(50, 50, 0, 0), minimum_goal_distance=5, render_mode="rgb_array")

for maze in sample_mazes(complexity_space, density_space):
    env.bit_maze = maze
    obs, _ = env.reset()
    display.display(image(env.render()))
