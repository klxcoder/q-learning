import numpy as np
import random
from numpy.typing import NDArray

class Maze:
    def __init__(self):
        self.maze = [
            ['S', '.', '.'],
            ['.', '#', '.'],
            ['.', '.', 'G'],
        ]
        self.rows = len(self.maze)
        self.cols = len(self.maze[0])
        self.start_pos: tuple[int, int] = (0, 0)
        self.goal_pos: tuple[int, int] = (2, 2)
        self.wall = '#'
        self.actions = ['up', 'down', 'left', 'right']
        self.action_space = len(self.actions)

    def get_state(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def get_pos(self, state: int) -> tuple[int, int]:
        return (state // self.cols, state % self.cols)

    def is_valid(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.maze[r][c] != self.wall

    def get_reward(self, pos: tuple[int, int]) -> float:
        if pos == self.goal_pos:
            return 1
        return 0

    def step(self, current_pos: tuple[int, int], action_index: int) -> tuple[tuple[int, int], float, bool]:
        action = self.actions[action_index]
        r, c = current_pos
        next_pos = None

        if action == 'up':
            next_pos = (r - 1, c)
        elif action == 'down':
            next_pos = (r + 1, c)
        elif action == 'left':
            next_pos = (r, c - 1)
        elif action == 'right':
            next_pos = (r, c + 1)

        if next_pos and self.is_valid(next_pos):
            reward = self.get_reward(next_pos)
            return next_pos, reward, next_pos == self.goal_pos
        else:
            return current_pos, -0.1, False # Small negative reward for hitting a wall or boundary

def q_learning(
        maze: Maze,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        episodes: int = 1000
    ) -> NDArray[np.float64]:

    q_table: NDArray[np.float64] = np.zeros((maze.rows * maze.cols, maze.action_space))

    for _ in range(episodes):
        current_pos = maze.start_pos
        done = False

        while not done:
            current_state: int = maze.get_state(current_pos)

            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action_index: int = random.choice(range(maze.action_space)) # Explore
            else:
                action_index: int = int(np.argmax(q_table[current_state, :])) # Exploit

            next_pos, reward, done = maze.step(current_pos, action_index)
            next_state = maze.get_state(next_pos)

            # Q-learning update rule
            q_table[current_state, action_index] += learning_rate * (
                reward + discount_factor * np.max(q_table[next_state, :]) - q_table[current_state, action_index]
            )

            current_pos = next_pos

    return q_table

def play_game(maze: Maze, q_table: NDArray[np.float64]) -> list[tuple[int, int]]:
    current_pos: tuple[int, int] = maze.start_pos
    done: bool = False
    path: list[tuple[int, int]] = [current_pos]

    while not done:
        current_state = maze.get_state(current_pos)
        action_index = int(np.argmax(q_table[current_state, :]))
        next_pos, _, done = maze.step(current_pos, action_index)
        path.append(next_pos)
        current_pos = next_pos
    return path

if __name__ == "__main__":
    maze = Maze()
    q_table: NDArray[np.float64] = q_learning(maze, episodes=1000)

    print("Q-Table:")
    print(q_table)

    print("\nOptimal Path:")
    optimal_path = play_game(maze, q_table)
    for row in range(maze.rows):
        line = ""
        for col in range(maze.cols):
            pos = (row, col)
            if pos in optimal_path:
                if pos == maze.start_pos:
                    line += "S "
                elif pos == maze.goal_pos:
                    line += "G "
                else:
                    line += "* "
            else:
                line += maze.maze[row][col] + " "
        print(line)