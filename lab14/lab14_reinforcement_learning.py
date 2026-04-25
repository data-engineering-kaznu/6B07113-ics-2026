from dataclasses import dataclass
from pathlib import Path
import csv
import random

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42


class BalanceBeamEnv:
    def __init__(self, max_position: int = 10, max_velocity: int = 3, max_steps: int = 200) -> None:
        self.max_position = max_position
        self.max_velocity = max_velocity
        self.max_steps = max_steps
        self.actions = [-1, 0, 1]
        self.position_values = list(range(-max_position, max_position + 1))
        self.velocity_values = list(range(-max_velocity, max_velocity + 1))
        self.state_count = len(self.position_values) * len(self.velocity_values)
        self.action_count = len(self.actions)
        self.position = 0
        self.velocity = 0
        self.steps = 0

    def state_to_index(self, position: int, velocity: int) -> int:
        position_idx = position + self.max_position
        velocity_idx = velocity + self.max_velocity
        return position_idx * len(self.velocity_values) + velocity_idx

    def index_to_state(self, index: int) -> tuple[int, int]:
        velocity_count = len(self.velocity_values)
        position_idx = index // velocity_count
        velocity_idx = index % velocity_count
        return position_idx - self.max_position, velocity_idx - self.max_velocity

    def reset(self) -> int:
        self.position = random.choice([-1, 0, 1])
        self.velocity = 0
        self.steps = 0
        return self.state_to_index(self.position, self.velocity)

    def step(self, action_index: int) -> tuple[int, float, bool]:
        action = self.actions[action_index]
        self.velocity += action
        self.velocity = int(np.clip(self.velocity, -self.max_velocity, self.max_velocity))
        self.position += self.velocity
        self.steps += 1

        out_of_bounds = abs(self.position) > self.max_position
        done = out_of_bounds or self.steps >= self.max_steps
        reward = -10.0 if out_of_bounds else 1.0

        self.position = int(np.clip(self.position, -self.max_position, self.max_position))
        next_state = self.state_to_index(self.position, self.velocity)
        return next_state, reward, done


@dataclass
class TrainingResult:
    rewards: list[float]
    episode_lengths: list[int]
    q_table: np.ndarray
    evaluation_lengths: list[int]


def epsilon_greedy(q_table: np.ndarray, state: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(q_table.shape[1])
    return int(np.argmax(q_table[state]))


def train_agent(
    env: BalanceBeamEnv,
    episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
) -> TrainingResult:
    q_table = np.zeros((env.state_count, env.action_count))
    rewards: list[float] = []
    episode_lengths: list[int] = []
    epsilon = epsilon_start

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = epsilon_greedy(q_table, state, epsilon)
            next_state, reward, done = env.step(action)
            best_next = np.max(q_table[next_state])

            q_table[state, action] += alpha * (
                reward + gamma * best_next * (0 if done else 1) - q_table[state, action]
            )

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        episode_lengths.append(steps)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    evaluation_lengths = evaluate_agent(env, q_table, episodes=20)
    return TrainingResult(rewards, episode_lengths, q_table, evaluation_lengths)


def evaluate_agent(env: BalanceBeamEnv, q_table: np.ndarray, episodes: int = 20) -> list[int]:
    lengths: list[int] = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            action = int(np.argmax(q_table[state]))
            state, _, done = env.step(action)
            steps += 1

        lengths.append(steps)

    return lengths


def moving_average(values: list[float], window: int = 50) -> np.ndarray:
    series = np.array(values, dtype=float)
    if len(series) < window:
        return series
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode="valid")


def plot_rewards(rewards: list[float], lengths: list[int], output_name: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(rewards, color="#4c78a8", alpha=0.5, label="Reward per episode")
    ma_rewards = moving_average(rewards, window=50)
    axes[0].plot(range(len(ma_rewards)), ma_rewards, color="#e45756", linewidth=2, label="Moving average (50)")
    axes[0].set_title("Q-learning Training Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lengths, color="#54a24b", alpha=0.6, label="Episode length")
    ma_lengths = moving_average(lengths, window=50)
    axes[1].plot(range(len(ma_lengths)), ma_lengths, color="#f58518", linewidth=2, label="Moving average (50)")
    axes[1].set_title("Episode Length During Training")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps survived")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def plot_q_table_heatmap(env: BalanceBeamEnv, q_table: np.ndarray, output_name: str) -> None:
    best_action = np.argmax(q_table, axis=1).reshape(len(env.position_values), len(env.velocity_values))
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(best_action, cmap="viridis", aspect="auto")
    ax.set_title("Best Action by State")
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Position")
    ax.set_xticks(range(len(env.velocity_values)))
    ax.set_xticklabels(env.velocity_values)
    ax.set_yticks(range(len(env.position_values)))
    ax.set_yticklabels(env.position_values)

    for i in range(best_action.shape[0]):
        for j in range(best_action.shape[1]):
            action_label = ["L", "S", "R"][best_action[i, j]]
            ax.text(j, i, action_label, ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def save_training_log(rewards: list[float], lengths: list[int], output_name: str) -> None:
    with (BASE_DIR / output_name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode", "reward", "steps_survived"])
        for idx, (reward, length) in enumerate(zip(rewards, lengths), start=1):
            writer.writerow([idx, f"{reward:.2f}", length])


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env = BalanceBeamEnv(max_position=10, max_velocity=3, max_steps=200)
    result = train_agent(env)

    average_last_100 = float(np.mean(result.episode_lengths[-100:]))
    best_episode = int(max(result.episode_lengths))
    evaluation_average = float(np.mean(result.evaluation_lengths))
    evaluation_best = int(max(result.evaluation_lengths))

    print("Lab 14: Reinforcement Learning")
    print("Environment: custom BalanceBeamEnv (CartPole-like educational environment)")
    print(f"State count: {env.state_count}")
    print(f"Action count: {env.action_count}")
    print(f"Training episodes: {len(result.rewards)}")
    print(f"Best training episode length: {best_episode}")
    print(f"Average episode length over last 100 episodes: {average_last_100:.2f}")
    print(f"Evaluation episode lengths: {result.evaluation_lengths}")
    print(f"Average evaluation length: {evaluation_average:.2f}")
    print(f"Best evaluation length: {evaluation_best}")

    if evaluation_best >= 200:
        print("Target reached: the agent can keep balance for at least 200 steps.")
    else:
        print("Target not reached: the agent did not reach 200 steps yet.")

    plot_rewards(result.rewards, result.episode_lengths, "training_rewards.png")
    plot_q_table_heatmap(env, result.q_table, "q_table_policy.png")
    save_training_log(result.rewards, result.episode_lengths, "training_log.csv")


if __name__ == "__main__":
    main()
