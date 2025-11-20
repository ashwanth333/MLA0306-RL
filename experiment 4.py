    import numpy as np

# ----------------------------------------------------------
# Grid World Environment
# ----------------------------------------------------------

class GridWorld:
    def __init__(self, rows, cols, warehouse, delivery_points, obstacles=None, step_cost=-1):
        self.rows = rows
        self.cols = cols
        self.warehouse = warehouse
        self.delivery_points = delivery_points
        self.obstacles = obstacles if obstacles else []
        self.step_cost = step_cost

        # Actions: up, down, left, right
        self.actions = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }

    def is_terminal(self, state):
        return state in self.delivery_points

    def get_next_state(self, state, action):
        if self.is_terminal(state) or state in self.obstacles:
            return state

        dr, dc = self.actions[action]
        r, c = state
        nr, nc = r + dr, c + dc

        # Check boundaries and obstacles
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.obstacles:
            return (nr, nc)
        return state

    def get_reward(self, state):
        if state in self.delivery_points:
            return 10
        return self.step_cost

# ----------------------------------------------------------
# Policy Iteration Algorithm
# ----------------------------------------------------------

def policy_iteration(env, discount=0.9, theta=1e-4):
    # Initialize policy randomly
    policy = {}
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) not in env.obstacles:
                policy[(r, c)] = np.random.choice(list(env.actions.keys()))

    V = {state: 0 for state in policy.keys()}

    stable = False
    while not stable:
        # -----------------------------
        # Policy Evaluation
        # -----------------------------
        while True:
            delta = 0
            for state in V:
                old_v = V[state]
                action = policy[state]
                next_state = env.get_next_state(state, action)
                reward = env.get_reward(next_state)

                V[state] = reward + discount * V.get(next_state, 0)
                delta = max(delta, abs(old_v - V[state]))
            if delta < theta:
                break

        # -----------------------------
        # Policy Improvement
        # -----------------------------
        stable = True

        for state in V:
            old_action = policy[state]
            best_value = float('-inf')
            best_action = old_action

            for action in env.actions:
                next_state = env.get_next_state(state, action)
                reward = env.get_reward(next_state)
                value = reward + discount * V.get(next_state, 0)

                if value > best_value:
                    best_value = value
                    best_action = action

            policy[state] = best_action

            if old_action != best_action:
                stable = False

    return policy, V

# ----------------------------------------------------------
# Example Usage
# ----------------------------------------------------------

if __name__ == "__main__":
    rows, cols = 6, 6
    warehouse = (0, 0)
    delivery_points = [(5, 5), (3, 4)]
    obstacles = [(1, 2), (2, 2), (3, 2)]

    env = GridWorld(rows, cols, warehouse, delivery_points, obstacles)

    policy, V = policy_iteration(env)

    print("\nOptimal Policy:")
    for r in range(rows):
        row = ""
        for c in range(cols):
            if (r, c) in obstacles:
                row += " X  "
            elif (r, c) in delivery_points:
                row += " D  "
            else:
                row += f" {policy[(r,c)]}  "
        print(row)

    print("\nValue Function:")
    for r in range(rows):
        row = ""
        for c in range(cols):
            if (r, c) in obstacles:
                row += "  X     "
            else:
                row += f"{V[(r,c)]:6.2f} "
        print(row)


