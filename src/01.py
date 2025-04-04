import numpy as np
import random

# Setup
states: list[str] = ['A', 'B', 'C']     # example states
actions: list[str] = ['left', 'right'] # example actions
q_table = np.zeros((len(states), len(actions)))  # Init Q-table

# Parameters
alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.2 # exploration rate

# Helper
def choose_action(state_idx: int):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, len(actions) - 1)  # explore
    return np.argmax(q_table[state_idx])            # exploit

# Example step
s: int = 0                      # current state index
a: (int | np.intp) = choose_action(s)       # choose action
r: float = 1                      # reward received
s_next: int = 1                 # next state index

# Q-table update
best_next: float = np.max(q_table[s_next])
q_table[s][a] += alpha * (r + gamma * best_next - q_table[s][a])

print(q_table)