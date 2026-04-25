from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_ACTIONS = 2
MAX_STEPS = 250

POSITION_BINS = np.linspace(-2.4, 2.4, 9)[1:-1]
VELOCITY_BINS = np.linspace(-3.0, 3.0, 9)[1:-1]
ANGLE_BINS = np.linspace(-0.21, 0.21, 11)[1:-1]
ANGULAR_VELOCITY_BINS = np.linspace(-3.5, 3.5, 11)[1:-1]


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool


class SimpleCartPoleEnv:
    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * np.pi / 180
        self.max_steps = MAX_STEPS

        self.rng = np.random.default_rng(random_state)
        self.state = np.zeros(4, dtype=float)
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.state = self.rng.uniform(low=-0.03, high=0.03, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> StepResult:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        theta_acc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=float)
        self.steps += 1

        done = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold_radians
            or self.steps >= self.max_steps
        )

        balance_bonus = 1.0 - 0.35 * abs(theta) / self.theta_threshold_radians
        centering_bonus = 1.0 - 0.15 * abs(x) / self.x_threshold
        reward = max(0.0, balance_bonus + centering_bonus)
        if done and self.steps < self.max_steps:
            reward = -2.0

        return StepResult(self.state.copy(), reward, done)


def discretize_state(state: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.digitize(state[0], POSITION_BINS)),
        int(np.digitize(state[1], VELOCITY_BINS)),
        int(np.digitize(state[2], ANGLE_BINS)),
        int(np.digitize(state[3], ANGULAR_VELOCITY_BINS)),
    )


def epsilon_by_episode(episode: int, total_episodes: int) -> float:
    progress = episode / total_episodes
    return max(0.03, (1.0 - progress) ** 1.8)


def alpha_by_episode(episode: int, total_episodes: int) -> float:
    progress = episode / total_episodes
    return max(0.08, 0.45 * (1.0 - progress) + 0.08)


def train_agent(
    env: SimpleCartPoleEnv,
    episodes: int = 2600,
    gamma: float = 0.99,
) -> tuple[np.ndarray, list[int], list[float]]:
    q_table = np.zeros(
        (
            len(POSITION_BINS) + 1,
            len(VELOCITY_BINS) + 1,
            len(ANGLE_BINS) + 1,
            len(ANGULAR_VELOCITY_BINS) + 1,
            N_ACTIONS,
        ),
        dtype=float,
    )
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []

    for episode in range(episodes):
        state = discretize_state(env.reset())
        epsilon = epsilon_by_episode(episode, episodes)
        alpha = alpha_by_episode(episode, episodes)
        total_reward = 0.0

        for step in range(1, env.max_steps + 1):
            if env.rng.random() < epsilon:
                action = int(env.rng.integers(0, N_ACTIONS))
            else:
                action = int(np.argmax(q_table[state]))

            result = env.step(action)
            next_state = discretize_state(result.state)
            best_next = np.max(q_table[next_state])
            td_target = result.reward + gamma * best_next * (not result.done)
            q_table[state + (action,)] += alpha * (td_target - q_table[state + (action,)])

            state = next_state
            total_reward += result.reward
            if result.done:
                episode_lengths.append(step)
                episode_rewards.append(total_reward)
                break

    return q_table, episode_lengths, episode_rewards


def evaluate_agent(
    env: SimpleCartPoleEnv,
    q_table: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    state_values: list[np.ndarray] = []
    state = env.reset()
    total_reward = 0.0

    for step in range(1, env.max_steps + 1):
        discrete_state = discretize_state(state)
        action = int(np.argmax(q_table[discrete_state]))
        state_values.append(state.copy())

        result = env.step(action)
        state = result.state
        total_reward += result.reward
        if result.done:
            state_values.append(state.copy())
            return step, total_reward, np.array(state_values)

    return env.max_steps, total_reward, np.array(state_values)


def moving_average(values: list[float], window: int = 50) -> np.ndarray:
    series = np.array(values, dtype=float)
    if series.size < window:
        return series
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(series, kernel, mode="valid")


def save_learning_curve(lengths: list[int], rewards: list[float], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(lengths, color="#2563eb", alpha=0.35, label="Episode length")
    axes[0].plot(
        np.arange(len(moving_average(lengths))) + 49,
        moving_average(lengths),
        color="#1d4ed8",
        linewidth=2,
        label="Moving average (50)",
    )
    axes[0].set_ylabel("Steps")
    axes[0].set_title("Q-learning training progress")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(rewards, color="#16a34a", alpha=0.35, label="Episode reward")
    axes[1].plot(
        np.arange(len(moving_average(rewards))) + 49,
        moving_average(rewards),
        color="#15803d",
        linewidth=2,
        label="Moving average (50)",
    )
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_policy_heatmap(q_table: np.ndarray, output_path: Path) -> None:
    position_index = len(POSITION_BINS) // 2
    velocity_index = len(VELOCITY_BINS) // 2
    policy_slice = np.argmax(q_table[position_index, velocity_index], axis=-1)

    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(policy_slice, origin="lower", cmap="coolwarm", aspect="auto")
    axis.set_title("Policy slice for angle and angular velocity")
    axis.set_xlabel("Angular velocity bin")
    axis.set_ylabel("Angle bin")
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["Left", "Right"])

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_demo_trajectory(trajectory: np.ndarray, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    steps = np.arange(len(trajectory))

    axes[0].plot(steps, trajectory[:, 0], label="Cart position", color="#7c3aed")
    axes[0].axhline(2.4, linestyle="--", color="#ef4444", alpha=0.6)
    axes[0].axhline(-2.4, linestyle="--", color="#ef4444", alpha=0.6)
    axes[0].set_ylabel("Position")
    axes[0].set_title("Greedy policy demo episode")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, trajectory[:, 2], label="Pole angle", color="#ea580c")
    axes[1].axhline(12 * np.pi / 180, linestyle="--", color="#ef4444", alpha=0.6)
    axes[1].axhline(-12 * np.pi / 180, linestyle="--", color="#ef4444", alpha=0.6)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Angle, rad")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    env = SimpleCartPoleEnv()
    q_table, lengths, rewards = train_agent(env)

    evaluation_env = SimpleCartPoleEnv(random_state=7)
    demo_steps, demo_reward, trajectory = evaluate_agent(evaluation_env, q_table)

    learning_curve_path = OUTPUT_DIR / "learning_curve.png"
    policy_path = OUTPUT_DIR / "policy_heatmap.png"
    trajectory_path = OUTPUT_DIR / "demo_trajectory.png"

    save_learning_curve(lengths, rewards, learning_curve_path)
    save_policy_heatmap(q_table, policy_path)
    save_demo_trajectory(trajectory, trajectory_path)

    best_episode = int(np.max(lengths))
    mean_last_100 = float(np.mean(lengths[-100:]))

    print("Lab 14. Reinforcement learning with Q-learning")
    print(f"Training episodes: {len(lengths)}")
    print(f"Best episode length: {best_episode}")
    print(f"Mean episode length over last 100 episodes: {mean_last_100:.2f}")
    print(f"Demo episode length: {demo_steps}")
    print(f"Demo episode reward: {demo_reward:.2f}")
    print(f"Learning curve saved to: {learning_curve_path.name}")
    print(f"Policy heatmap saved to: {policy_path.name}")
    print(f"Demo trajectory saved to: {trajectory_path.name}")


if __name__ == "__main__":
    main()
