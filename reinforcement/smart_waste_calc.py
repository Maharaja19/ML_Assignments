import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------
# Environment Setup
# -----------------------------

GRID_SIZE = 5  # 5x5 city grid
NUM_STATES = GRID_SIZE * GRID_SIZE
NUM_ACTIONS = 4  # up, down, left, right

# Actions
ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1)    # right
}

# Random bin fill levels (0 to 1)
bin_levels = np.random.rand(GRID_SIZE, GRID_SIZE)

# Threshold for "full"
FULL_THRESHOLD = 0.7


def state_to_pos(state):
    return (state // GRID_SIZE, state % GRID_SIZE)


def pos_to_state(pos):
    return pos[0] * GRID_SIZE + pos[1]


def step(state, action):
    row, col = state_to_pos(state)
    dr, dc = ACTIONS[action]

    new_row = min(max(row + dr, 0), GRID_SIZE - 1)
    new_col = min(max(col + dc, 0), GRID_SIZE - 1)

    next_state = pos_to_state((new_row, new_col))

    # Reward logic
    if bin_levels[new_row][new_col] > FULL_THRESHOLD:
        reward = 10   # good: full bin
    else:
        reward = -2   # waste of time

    return next_state, reward


# -----------------------------
# SARSA Parameters
# -----------------------------

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

Q = np.zeros((NUM_STATES, NUM_ACTIONS))


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        return np.argmax(Q[state])


# -----------------------------
# Training
# -----------------------------

rewards_per_episode = []

for ep in range(episodes):
    state = random.randint(0, NUM_STATES - 1)
    action = choose_action(state)

    total_reward = 0

    for _ in range(50):
        next_state, reward = step(state, action)
        next_action = choose_action(next_state)

        # SARSA Update
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )

        state = next_state
        action = next_action
        total_reward += reward

    rewards_per_episode.append(total_reward)

print("Training Completed!")

# -----------------------------
# Plot Results
# -----------------------------

plt.plot(rewards_per_episode)
plt.title("Rewards per Episode")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.show()


# -----------------------------
# Test Policy
# -----------------------------

print("\nTesting Learned Policy:\n")

state = 0

for step_num in range(15):
    action = np.argmax(Q[state])
    next_state, reward = step(state, action)

    print(f"Step {step_num+1}: State={state}, Action={action}, Next={next_state}, Reward={reward}")

    state = next_state