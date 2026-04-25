import random

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


def make_env():
    return gym.make("CartPole-v1")


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, info = result
        return state, reward, terminated or truncated, info
    return result


def get_state(observation, bins):
    state = []
    for value, borders in zip(observation, bins):
        state.append(np.digitize(value, borders))
    return tuple(state)


def choose_action(q_table, state, epsilon, action_count):
    if random.random() < epsilon:
        return random.randrange(action_count)
    return int(np.argmax(q_table[state]))


def train():
    env = make_env()

    bins = [
        np.linspace(-2.4, 2.4, 9),
        np.linspace(-3.0, 3.0, 9),
        np.linspace(-0.5, 0.5, 9),
        np.linspace(-2.0, 2.0, 9),
    ]

    action_count = env.action_space.n
    q_table = np.zeros((10, 10, 10, 10, action_count))

    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.995
    episodes = 3000

    best_steps = 0

    for episode in range(episodes):
        observation = reset_env(env)
        state = get_state(observation, bins)
        done = False
        steps = 0

        while not done and steps < 500:
            action = choose_action(q_table, state, epsilon, action_count)
            new_observation, reward, done, _ = step_env(env, action)
            new_state = get_state(new_observation, bins)

            old_value = q_table[state + (action,)]
            next_value = np.max(q_table[new_state])
            q_table[state + (action,)] = old_value + alpha * (reward + gamma * next_value - old_value)

            state = new_state
            steps += 1

        best_steps = max(best_steps, steps)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 300 == 0:
            print(f"Episode {episode + 1}: steps = {steps}, best = {best_steps}")

    env.close()
    return q_table, bins


def test_agent(q_table, bins):
    env = make_env()
    observation = reset_env(env)
    state = get_state(observation, bins)
    done = False
    steps = 0

    while not done and steps < 500:
        action = int(np.argmax(q_table[state]))
        observation, _, done, _ = step_env(env, action)
        state = get_state(observation, bins)
        steps += 1

    env.close()
    return steps


def main():
    q_table, bins = train()
    steps = test_agent(q_table, bins)
    print(f"Test result: the agent balanced the pole for {steps} steps")


if __name__ == "__main__":
    main()
