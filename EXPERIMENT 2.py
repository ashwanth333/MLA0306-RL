import numpy as np

# ---------------------------------------------------------
# Warehouse Environment (Grid MDP)
# ---------------------------------------------------------

class WarehouseMDP:
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols

        # Define obstacles, items, and goal positions
        self.obstacles = {(1, 1), (3, 2)}
        self.items = {(0, 3), (2, 4)}
        self.goal = (4, 4)

        # Rewards
        self.R_item = 2
        self.R_goal = 5
        self.R_obstacle = -2
        self.R_step = 0  # normal movement

        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    def step(self, state, action):
        (x, y) = state

        if action == "UP":
            nx, ny = max(0, x - 1), y
        elif action == "DOWN":
            nx, ny = min(self.rows - 1, x + 1), y
        elif action == "LEFT":
            nx, ny = x, max(0, y - 1)
        elif action == "RIGHT":
            nx, ny = x, min(self.cols - 1, y + 1)
        else:
            nx, ny = x, y

        next_state = (nx, ny)

        # Assign reward
        if next_state in self.obstacles:
            reward = self.R_obstacle
        elif next_state in self.items:
            reward = self.R_item
        elif next_state == self.goal:
            reward = self.R_goal
        else:
            reward = self.R_step

        return next_state, reward


# ---------------------------------------------------------
# POLICY EVALUATION
# ---------------------------------------------------------

def policy_evaluation(env, policy, gamma=0.9, theta=1e-4):
    """
    env: WarehouseMDP()
    policy: dict mapping state -> action
    """
    # Initialize value function
    V = {(i, j): 0.0 for i in range(env.rows) for j in range(env.cols)}

    while True:
        delta = 0
        new_V = V.copy()

        for state in V:
            action = policy[state]                      # follow given deterministic policy
            next_state, reward = env.step(state, action)

            new_V[state] = reward + gamma * V[next_state]
            delta = max(delta, abs(new_V[state] - V[state]))

        V = new_V

        if delta < theta:
            break

    return V


# ---------------------------------------------------------
# EXAMPLE POLICY (moves right until last column, then down)
# ---------------------------------------------------------

def sample_policy(env):
    policy = {}
    for i in range(env.rows):
        for j in range(env.cols):
            if j < env.cols - 1:
                policy[(i, j)] = "RIGHT"
            else:
                policy[(i, j)] = "DOWN"
    return policy


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------
