import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "input10.csv"
OUTPUT_FILE = BASE_DIR / "output10.csv"

STATES = ["Sunny", "Rainy"]
START_PROBABILITIES = {
    "Sunny": 0.6,
    "Rainy": 0.4,
}
TRANSITION_PROBABILITIES = {
    "Sunny": {"Sunny": 0.7, "Rainy": 0.3},
    "Rainy": {"Sunny": 0.4, "Rainy": 0.6},
}
EMISSION_PROBABILITIES = {
    "Sunny": {"walk": 0.6, "shop": 0.3, "clean": 0.1},
    "Rainy": {"walk": 0.1, "shop": 0.4, "clean": 0.5},
}


def load_sequences(filename):
    sequences = []
    with filename.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            observations = row["observations"].strip().split()
            sequences.append(
                {
                    "sequence_id": row["sequence_id"],
                    "observations": observations,
                }
            )
    return sequences


def validate_observations(observations):
    allowed_observations = set(EMISSION_PROBABILITIES["Sunny"].keys())
    invalid_values = [value for value in observations if value not in allowed_observations]
    if invalid_values:
        raise ValueError(
            "Unsupported observations found: " + ", ".join(sorted(set(invalid_values)))
        )


def viterbi(observations):
    validate_observations(observations)

    probabilities = []
    paths = {}

    first_step = {}
    for state in STATES:
        first_step[state] = (
            START_PROBABILITIES[state] * EMISSION_PROBABILITIES[state][observations[0]]
        )
        paths[state] = [state]
    probabilities.append(first_step)

    for index in range(1, len(observations)):
        current_step = {}
        new_paths = {}
        current_observation = observations[index]

        for current_state in STATES:
            best_probability = -1.0
            best_previous_state = None

            for previous_state in STATES:
                probability = (
                    probabilities[index - 1][previous_state]
                    * TRANSITION_PROBABILITIES[previous_state][current_state]
                    * EMISSION_PROBABILITIES[current_state][current_observation]
                )
                if probability > best_probability:
                    best_probability = probability
                    best_previous_state = previous_state

            current_step[current_state] = best_probability
            new_paths[current_state] = paths[best_previous_state] + [current_state]

        probabilities.append(current_step)
        paths = new_paths

    final_state = max(probabilities[-1], key=probabilities[-1].get)
    return paths[final_state], probabilities[-1][final_state]


def build_transition_rows():
    rows = []
    for from_state in STATES:
        for to_state in STATES:
            probability = TRANSITION_PROBABILITIES[from_state][to_state]
            bar = "#" * int(round(probability * 20))
            rows.append(
                [
                    "transition",
                    f"{from_state}->{to_state}",
                    "",
                    "",
                    f"{probability:.2f}",
                    bar,
                ]
            )
    return rows


def build_result_rows(sequences):
    rows = []
    for item in sequences:
        best_path, final_probability = viterbi(item["observations"])
        rows.append(
            [
                "sequence_result",
                item["sequence_id"],
                " ".join(item["observations"]),
                " ".join(best_path),
                f"{final_probability:.8f}",
                "",
            ]
        )
    return rows


def save_output(filename, rows):
    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "record_type",
                "name",
                "observations",
                "best_states",
                "probability",
                "visualization",
            ]
        )
        writer.writerows(rows)


def main():
    sequences = load_sequences(INPUT_FILE)
    rows = build_transition_rows() + build_result_rows(sequences)
    save_output(OUTPUT_FILE, rows)
    print("Results saved to output10.csv")


if __name__ == "__main__":
    main()
