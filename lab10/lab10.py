from math import log

def safe_log(value: float) -> float:
    if value <= 0:
        return float("-inf")
    return log(value)


def viterbi(observations, states, start_prob, transition_prob, emission_prob):
    v = [{}]
    paths = {}

    for state in states:
        v[0][state] = safe_log(start_prob[state]) + safe_log(emission_prob[state][observations[0]])
        paths[state] = [state]

    for t in range(1, len(observations)):
        v.append({})
        new_paths = {}

        for current_state in states:
            best_prev_state = None
            best_score = float("-inf")

            for previous_state in states:
                score = (
                    v[t - 1][previous_state]
                    + safe_log(transition_prob[previous_state][current_state])
                    + safe_log(emission_prob[current_state][observations[t]])
                )
                if score > best_score:
                    best_score = score
                    best_prev_state = previous_state

            v[t][current_state] = best_score
            new_paths[current_state] = paths[best_prev_state] + [current_state]

        paths = new_paths

    best_final_state = max(v[-1], key=v[-1].get)
    return paths[best_final_state], v


def print_transition_matrix(states, transition_prob):
    print("Матрица вероятностей переходов:")
    header = " " * 12 + "".join(f"{state:>12}" for state in states)
    print(header)

    for from_state in states:
        row = f"{from_state:12}"
        for to_state in states:
            probability = transition_prob[from_state][to_state]
            row += f"{probability:12.2f}"
        print(row)


def print_viterbi_table(observations, states, v):
    print("\nТаблица Витерби (логарифмы вероятностей):")
    header = "Состояние".ljust(12) + "".join(f"{obs:>12}" for obs in observations)
    print(header)

    for state in states:
        row = state.ljust(12)
        for t in range(len(observations)):
            row += f"{v[t][state]:12.4f}"
        print(row)


def main():
    states = ["Sunny", "Rainy"]
    observations = ["walk", "shop", "clean"]

    start_prob = {
        "Sunny": 0.6,
        "Rainy": 0.4,
    }

    transition_prob = {
        "Sunny": {"Sunny": 0.7, "Rainy": 0.3},
        "Rainy": {"Sunny": 0.4, "Rainy": 0.6},
    }

    emission_prob = {
        "Sunny": {"walk": 0.6, "shop": 0.3, "clean": 0.1},
        "Rainy": {"walk": 0.1, "shop": 0.4, "clean": 0.5},
    }

    best_path, v = viterbi(
        observations,
        states,
        start_prob,
        transition_prob,
        emission_prob,
    )

    print("Наблюдения:", observations)
    print("Наиболее вероятная последовательность состояний:", " -> ".join(best_path))
    print_transition_matrix(states, transition_prob)
    print_viterbi_table(observations, states, v)


if __name__ == "__main__":
    main()
