import numpy as np
import random

# Environment setup
maze: list[list[str]] = [
    ['S', '.', '.'],
    ['.', '#', '.'],
    ['.', '.', 'G']
]

rows, cols = 3, 3
actions: list[str] = ['up', 'down', 'left', 'right']
q_table = np.zeros((rows, cols, len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

def is_valid(r: int, c: int):
    return 0 <= r < rows and 0 <= c < cols and maze[r][c] != '#'

def get_next_pos(r: int, c: int, action: str):
    r_new, c_new = r, c
    if action == 'up':    r_new -= 1
    if action == 'down':  r_new += 1
    if action == 'left':  c_new -= 1
    if action == 'right': c_new += 1
    return (r_new, c_new) if is_valid(r_new, c_new) else (r, c)

def choose_action(r: int, c: int):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, len(actions) - 1)
    return np.argmax(q_table[r][c])

episodes = 500

for _ in range(episodes):
    r, c = 0, 0  # Start at 'S'
    while maze[r][c] != 'G':
        a_idx = choose_action(r, c)
        a = actions[a_idx]
        r_next, c_next = get_next_pos(r, c, a)
        reward = 1 if maze[r_next][c_next] == 'G' else 0
        q_table[r][c][a_idx] += alpha * (
            reward + gamma * np.max(q_table[r_next][c_next]) - q_table[r][c][a_idx]
        )
        r, c = r_next, c_next

# After training: Predict optimal path from 'S' to 'G'
r, c = 0, 0
path = [(r, c)]
while maze[r][c] != 'G':
    a_idx = np.argmax(q_table[r][c])
    a = actions[a_idx]
    r, c = get_next_pos(r, c, a)
    path.append((r, c))
    if len(path) > 20: break  # avoid infinite loops if Q-table is bad

print(q_table)

print("Predicted path:", path)