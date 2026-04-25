import csv
import random
from pathlib import Path


INPUT_FILE = "input14.csv"
OUTPUT_FILE = "output14.csv"
REQUIRED_FIELDS = {"states", "goal_state", "episodes"}


def read_config(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            raise ValueError(
                f"Файл {path.name} должен содержать столбцы: states, goal_state, episodes."
            )

        row = next(reader, None)
        if row is None:
            raise ValueError(f"Файл {path.name} не содержит настроек среды.")

        try:
            states = int(row["states"])
            goal_state = int(row["goal_state"])
            episodes = int(row["episodes"])
        except (TypeError, ValueError) as error:
            raise ValueError("Параметры среды должны быть целыми числами.") from error

        if states < 2:
            raise ValueError("Количество состояний должно быть не меньше 2.")
        if not 0 <= goal_state < states:
            raise ValueError("Целевое состояние должно находиться в диапазоне состояний.")
        if episodes < 1:
            raise ValueError("Количество эпизодов должно быть положительным.")

        return {
            "states": states,
            "goal_state": goal_state,
            "episodes": episodes,
        }


class LineWorld:
    ACTIONS = ("left", "right")

    def __init__(self, states, goal_state):
        self.states = states
        self.goal_state = goal_state
        self.start_state = 0 if goal_state != 0 else states - 1

    def reset(self):
        return self.start_state

    def step(self, state, action_index):
        if action_index == 0:
            next_state = max(0, state - 1)
        else:
            next_state = min(self.states - 1, state + 1)

        done = next_state == self.goal_state
        reward = 10 if done else -1
        return next_state, reward, done


def train_agent(env, episodes, alpha=0.2, gamma=0.9, epsilon=0.2, seed=42):
    random.seed(seed)
    q_table = [[0.0 for _ in env.ACTIONS] for _ in range(env.states)]

    for _ in range(episodes):
        state = env.reset()
        for _ in range(env.states * 3):
            if random.random() < epsilon:
                action = random.randrange(len(env.ACTIONS))
            else:
                action = 0 if q_table[state][0] >= q_table[state][1] else 1

            next_state, reward, done = env.step(state, action)
            best_next = max(q_table[next_state])
            q_table[state][action] += alpha * (
                reward + gamma * best_next - q_table[state][action]
            )
            state = next_state

            if done:
                break

    return q_table


def greedy_path(env, q_table):
    state = env.reset()
    path = [state]
    visited = {state}

    for _ in range(env.states * 2):
        if state == env.goal_state:
            break

        action = 0 if q_table[state][0] >= q_table[state][1] else 1
        next_state, _, done = env.step(state, action)
        path.append(next_state)

        if done or next_state in visited:
            break

        visited.add(next_state)
        state = next_state

    success = path[-1] == env.goal_state
    return path, success


def write_results(path, q_table, env, path_states, success):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["state", "q_left", "q_right", "best_action"])
        for state, values in enumerate(q_table):
            best_action = env.ACTIONS[0] if values[0] >= values[1] else env.ACTIONS[1]
            writer.writerow(
                [state, f"{values[0]:.6f}", f"{values[1]:.6f}", best_action]
            )

        writer.writerow([])
        writer.writerow(["path", "success"])
        writer.writerow([" -> ".join(map(str, path_states)), "yes" if success else "no"])


def main():
    base_dir = Path(__file__).resolve().parent
    config = read_config(base_dir / INPUT_FILE)

    env = LineWorld(config["states"], config["goal_state"])
    q_table = train_agent(env, config["episodes"])
    path_states, success = greedy_path(env, q_table)
    write_results(base_dir / OUTPUT_FILE, q_table, env, path_states, success)

    print("Обучение завершено. Результаты сохранены в output14.csv.")
    print(f"Путь агента до цели: {' -> '.join(map(str, path_states))}.")


if __name__ == "__main__":
    main()
