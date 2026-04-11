from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_hmm_dataset.csv"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def estimate_hmm(train_df: pd.DataFrame, states: list[str], observations: list[str], alpha: float = 1.0):
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    obs_to_idx = {obs: idx for idx, obs in enumerate(observations)}

    start_counts = np.full(len(states), alpha, dtype=float)
    transition_counts = np.full((len(states), len(states)), alpha, dtype=float)
    emission_counts = np.full((len(states), len(observations)), alpha, dtype=float)

    for _, group in train_df.groupby("sequence_id"):
        sequence = group.sort_values("step")
        hidden_states = sequence["weather"].tolist()
        emitted_observations = sequence["activity"].tolist()

        start_counts[state_to_idx[hidden_states[0]]] += 1

        for state, obs in zip(hidden_states, emitted_observations):
            emission_counts[state_to_idx[state], obs_to_idx[obs]] += 1

        for current_state, next_state in zip(hidden_states[:-1], hidden_states[1:]):
            transition_counts[state_to_idx[current_state], state_to_idx[next_state]] += 1

    start_probs = start_counts / start_counts.sum()
    transition_probs = normalize_rows(transition_counts)
    emission_probs = normalize_rows(emission_counts)

    return start_probs, transition_probs, emission_probs


def viterbi(observation_sequence: list[str], states: list[str], observations: list[str], start_probs, transition_probs, emission_probs):
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    obs_to_idx = {obs: idx for idx, obs in enumerate(observations)}

    n_states = len(states)
    n_steps = len(observation_sequence)

    log_start = np.log(start_probs)
    log_transition = np.log(transition_probs)
    log_emission = np.log(emission_probs)

    dp = np.full((n_states, n_steps), -np.inf)
    backpointer = np.zeros((n_states, n_steps), dtype=int)

    first_obs = obs_to_idx[observation_sequence[0]]
    for state_idx in range(n_states):
        dp[state_idx, 0] = log_start[state_idx] + log_emission[state_idx, first_obs]

    for step in range(1, n_steps):
        obs_idx = obs_to_idx[observation_sequence[step]]
        for current_state in range(n_states):
            scores = dp[:, step - 1] + log_transition[:, current_state]
            best_prev_state = int(np.argmax(scores))
            dp[current_state, step] = scores[best_prev_state] + log_emission[current_state, obs_idx]
            backpointer[current_state, step] = best_prev_state

    best_last_state = int(np.argmax(dp[:, -1]))
    best_path = [best_last_state]

    for step in range(n_steps - 1, 0, -1):
        best_last_state = backpointer[best_last_state, step]
        best_path.append(best_last_state)

    best_path.reverse()
    decoded_states = [states[idx] for idx in best_path]
    best_log_probability = float(np.max(dp[:, -1]))

    return decoded_states, best_log_probability


def plot_matrix(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, output_name: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    ax.set_xlabel("To")
    ax.set_ylabel("From")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["sequence_id", "step"]).reset_index(drop=True)

    states = sorted(df["weather"].unique().tolist())
    observations = sorted(df["activity"].unique().tolist())

    train_df = df[df["sequence_id"] < 5].copy()
    test_df = df[df["sequence_id"] == 5].copy()

    start_probs, transition_probs, emission_probs = estimate_hmm(train_df, states, observations)

    test_observations = test_df["activity"].tolist()
    true_states = test_df["weather"].tolist()
    predicted_states, log_probability = viterbi(
        test_observations,
        states,
        observations,
        start_probs,
        transition_probs,
        emission_probs,
    )

    accuracy = np.mean(np.array(predicted_states) == np.array(true_states))

    print("Lab 10: Hidden Markov Model")
    print(f"Dataset rows: {len(df)}")
    print(f"Training sequences: {train_df['sequence_id'].nunique()}")
    print(f"Test sequence length: {len(test_df)}")
    print()
    print("States:", states)
    print("Observations:", observations)
    print()
    print("Start probabilities:")
    for state, prob in zip(states, start_probs):
        print(f"  {state}: {prob:.3f}")
    print()
    print("Test observations:")
    print("  " + " -> ".join(test_observations))
    print("True states:")
    print("  " + " -> ".join(true_states))
    print("Predicted states (Viterbi):")
    print("  " + " -> ".join(predicted_states))
    print(f"Sequence accuracy: {accuracy:.3f}")
    print(f"Log-probability of best path: {log_probability:.3f}")

    transition_df = pd.DataFrame(transition_probs, index=states, columns=states)
    emission_df = pd.DataFrame(emission_probs, index=states, columns=observations)

    print()
    print("Transition probability matrix:")
    print(transition_df.round(3))
    print()
    print("Emission probability matrix:")
    print(emission_df.round(3))

    plot_matrix(
        transition_probs,
        states,
        states,
        "HMM Transition Probabilities",
        "transition_probabilities.png",
    )
    plot_matrix(
        emission_probs,
        states,
        observations,
        "HMM Emission Probabilities",
        "emission_probabilities.png",
    )


if __name__ == "__main__":
    main()
