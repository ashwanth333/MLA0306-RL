import numpy as np
import random
import math

# ---------------------------------------------------------
# PRICE ARMS (each arm = price level)
# ---------------------------------------------------------
prices = [50, 70, 90, 110]  

# Hidden conversion probabilities for each price
# (In real world, retailer does NOT know these)
true_conversion = [0.30, 0.22, 0.15, 0.10]


# ---------------------------------------------------------
# REVENUE SIMULATION: pulls an arm and returns revenue
# ---------------------------------------------------------
def pull_arm(arm):
    price = prices[arm]
    prob = true_conversion[arm]
    sale = 1 if random.random() < prob else 0
    return price * sale


# ---------------------------------------------------------
# EPSILON-GREEDY BANDIT
# ---------------------------------------------------------
def epsilon_greedy(T=1000, epsilon=0.1):
    n_arms = len(prices)
    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    revenue = 0

    for t in range(T):
        if random.random() < epsilon:
            arm = random.randint(0, n_arms - 1)  # explore
        else:
            arm = np.argmax(values)              # exploit

        r = pull_arm(arm)
        revenue += r
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]

    return revenue


# ---------------------------------------------------------
# UCB1 BANDIT
# ---------------------------------------------------------
def ucb(T=1000):
    n_arms = len(prices)
    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    revenue = 0

    # play each arm once
    for arm in range(n_arms):
        r = pull_arm(arm)
        revenue += r
        counts[arm] = 1
        values[arm] = r

    for t in range(n_arms, T):
        ucb_values = values + np.sqrt(2 * np.log(t + 1) / counts)
        arm = np.argmax(ucb_values)

        r = pull_arm(arm)
        revenue += r
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]

    return revenue


# ---------------------------------------------------------
# THOMPSON SAMPLING BANDIT
# ---------------------------------------------------------
def thompson_sampling(T=1000):
    n_arms = len(prices)

    # Beta distribution parameters
    alpha = np.ones(n_arms)
    beta = np.ones(n_arms)

    revenue = 0

    for t in range(T):
        samples = [np.random.beta(alpha[i], beta[i]) for i in range(n_arms)]
        arm = np.argmax(samples)

        r = pull_arm(arm)
        revenue += r

        # binary: sale or no sale
        sale = 1 if r > 0 else 0
        if sale:
            alpha[arm] += 1
        else:
            beta[arm] += 1

    return revenue


# ---------------------------------------------------------
# RUN COMPARISON
# ---------------------------------------------------------
if __name__ == "__main__":
    T = 2000  # number of pricing decisions

    rev_eps = epsilon_greedy(T, epsilon=0.1)
    rev_ucb = ucb(T)
    rev_ts = thompson_sampling(T)

    print("\n--- TOTAL REVENUE AFTER", T, "ROUNDS ---")
    print("Epsilon-Greedy :", rev_eps)
    print("UCB1           :", rev_ucb)
    print("Thompson Samp. :", rev_ts)

    best = max(rev_eps, rev_ucb, rev_ts)
    if best == rev_eps:
        print("\nWinner: Epsilon-Greedy")
    elif best == rev_ucb:
        print("\nWinner: UCB1")
    else:
        print("\nWinner: Thompson Sampling")
