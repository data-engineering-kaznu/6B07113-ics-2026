from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = SCRIPT_DIR / ".cache"
LOCAL_MPL_DIR = SCRIPT_DIR / ".matplotlib"

LOCAL_CACHE_DIR.mkdir(exist_ok=True)
LOCAL_MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPL_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUTPUT_DIR = SCRIPT_DIR / "outputs"

STATES = ["Sunny", "Rainy"]
OBSERVATIONS = ["walk", "shop", "clean"]

START_PROBABILITY = {
    "Sunny": 0.6,
    "Rainy": 0.4,
}

TRANSITION_PROBABILITY = {
    "Sunny": {"Sunny": 0.7, "Rainy": 0.3},
    "Rainy": {"Sunny": 0.4, "Rainy": 0.6},
}

EMISSION_PROBABILITY = {
    "Sunny": {"walk": 0.6, "shop": 0.3, "clean": 0.1},
    "Rainy": {"walk": 0.1, "shop": 0.4, "clean": 0.5},
}


def viterbi(observations: list[str]) -> tuple[list[str], float, list[dict[str, float]]]:
    probabilities: list[dict[str, float]] = []
    paths: dict[str, list[str]] = {}

    first_step: dict[str, float] = {}
    for state in STATES:
        first_step[state] = (
            START_PROBABILITY[state] * EMISSION_PROBABILITY[state][observations[0]]
        )
        paths[state] = [state]
    probabilities.append(first_step)

    for step in range(1, len(observations)):
        current_probabilities: dict[str, float] = {}
        current_paths: dict[str, list[str]] = {}

        for current_state in STATES:
            candidates = []
            for previous_state in STATES:
                probability = (
                    probabilities[step - 1][previous_state]
                    * TRANSITION_PROBABILITY[previous_state][current_state]
                    * EMISSION_PROBABILITY[current_state][observations[step]]
                )
                candidates.append((probability, previous_state))

            best_probability, best_previous_state = max(candidates)
            current_probabilities[current_state] = best_probability
            current_paths[current_state] = paths[best_previous_state] + [current_state]

        probabilities.append(current_probabilities)
        paths = current_paths

    best_state = max(STATES, key=lambda state: probabilities[-1][state])
    return paths[best_state], probabilities[-1][best_state], probabilities


def plot_transition_matrix() -> None:
    matrix = [
        [TRANSITION_PROBABILITY[row_state][column_state] for column_state in STATES]
        for row_state in STATES
    ]

    plt.figure(figsize=(6, 5))
    image = plt.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(image, label="Probability")
    plt.xticks(range(len(STATES)), STATES)
    plt.yticks(range(len(STATES)), STATES)
    plt.xlabel("Next state")
    plt.ylabel("Current state")
    plt.title("HMM Transition Probabilities")

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            plt.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=12,
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transition_matrix.png", dpi=150)
    plt.close()


def plot_viterbi_probabilities(
    observations: list[str],
    probabilities: list[dict[str, float]],
) -> None:
    plt.figure(figsize=(8, 5))

    steps = list(range(1, len(observations) + 1))
    for state in STATES:
        state_probabilities = [step_probability[state] for step_probability in probabilities]
        plt.plot(steps, state_probabilities, marker="o", label=state)

    plt.xticks(steps, observations)
    plt.xlabel("Observation")
    plt.ylabel("Viterbi probability")
    plt.title("State Probabilities During Viterbi Decoding")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "viterbi_probabilities.png", dpi=150)
    plt.close()


def print_model() -> None:
    print("Lab 10: Hidden Markov Model for Weather Prediction")
    print("-" * 64)
    print("States:", ", ".join(STATES))
    print("Observations:", ", ".join(OBSERVATIONS))
    print()
    print("Transition probabilities:")
    for state, transitions in TRANSITION_PROBABILITY.items():
        values = ", ".join(
            f"{next_state}={probability:.2f}"
            for next_state, probability in transitions.items()
        )
        print(f"  {state}: {values}")
    print()
    print("Emission probabilities:")
    for state, emissions in EMISSION_PROBABILITY.items():
        values = ", ".join(
            f"{observation}={probability:.2f}"
            for observation, probability in emissions.items()
        )
        print(f"  {state}: {values}")


def print_result(
    observations: list[str],
    path: list[str],
    probability: float,
    probabilities: list[dict[str, float]],
) -> None:
    print()
    print("Input observation sequence:", " -> ".join(observations))
    print("Most probable hidden state sequence:", " -> ".join(path))
    print(f"Sequence probability: {probability:.8f}")
    print()
    print(f"{'Step':>4} | {'Observation':>11} | {'Sunny':>12} | {'Rainy':>12}")
    print("-" * 51)
    for index, observation in enumerate(observations):
        print(
            f"{index + 1:>4} | "
            f"{observation:>11} | "
            f"{probabilities[index]['Sunny']:>12.8f} | "
            f"{probabilities[index]['Rainy']:>12.8f}"
        )
    print()
    print(f"Plots saved to: {OUTPUT_DIR}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    observations = ["walk", "shop", "clean", "clean", "walk"]
    path, probability, probabilities = viterbi(observations)

    print_model()
    print_result(observations, path, probability, probabilities)
    plot_transition_matrix()
    plot_viterbi_probabilities(observations, probabilities)


if __name__ == "__main__":
    main()
