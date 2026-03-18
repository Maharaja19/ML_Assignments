import gym
import numpy as np

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Initialize Q-table
state_size = env.observation_space.n
action_size = env.action_space.n

Q = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.8      # Learning rate
gamma = 0.95     # Discount factor
epsilon = 0.1    # Exploration rate
episodes = 1000

# Training
for episode in range(episodes):
    state = env.reset()[0]
    done = False

    while not done:

        # Exploration vs Exploitation
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, _, _ = env.step(action)

        # Q-learning update formula
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

print("Training Completed!")
print("\nFinal Q-Table:")
print(Q)