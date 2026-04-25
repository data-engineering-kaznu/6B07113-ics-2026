import csv
import math
import random
from pathlib import Path


INPUT_FILE = "input13.csv"
OUTPUT_FILE = "output13.csv"
REQUIRED_FIELDS = {"x1", "x2", "label"}


def sigmoid(value):
    if value >= 0:
        exp_value = math.exp(-value)
        return 1 / (1 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            raise ValueError(
                f"Файл {path.name} должен содержать столбцы: x1, x2, label."
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            try:
                x1 = float(row["x1"])
                x2 = float(row["x2"])
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Некорректные данные в строке {line_number}: {row}"
                ) from error

            if label not in (0, 1):
                raise ValueError(
                    f"Метка в строке {line_number} должна быть 0 или 1."
                )

            rows.append(([x1, x2], label))

        if not rows:
            raise ValueError(f"Файл {path.name} не содержит данных для обработки.")

        return rows


class SimpleMLP:
    def __init__(self, input_size=2, hidden_size=4, learning_rate=0.7, seed=42):
        random.seed(seed)
        self.learning_rate = learning_rate
        self.hidden_weights = [
            [random.uniform(-1.0, 1.0) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.hidden_biases = [random.uniform(-1.0, 1.0) for _ in range(hidden_size)]
        self.output_weights = [random.uniform(-1.0, 1.0) for _ in range(hidden_size)]
        self.output_bias = random.uniform(-1.0, 1.0)

    def forward(self, inputs):
        hidden_outputs = []
        for weights, bias in zip(self.hidden_weights, self.hidden_biases):
            activation = sum(weight * value for weight, value in zip(weights, inputs)) + bias
            hidden_outputs.append(sigmoid(activation))

        output_activation = (
            sum(weight * value for weight, value in zip(self.output_weights, hidden_outputs))
            + self.output_bias
        )
        output = sigmoid(output_activation)
        return hidden_outputs, output

    def train(self, dataset, epochs=5000):
        for _ in range(epochs):
            for inputs, expected in dataset:
                hidden_outputs, output = self.forward(inputs)
                output_error = output - expected
                output_delta = output_error * output * (1 - output)

                previous_output_weights = self.output_weights[:]
                for index, hidden_value in enumerate(hidden_outputs):
                    self.output_weights[index] -= self.learning_rate * output_delta * hidden_value
                self.output_bias -= self.learning_rate * output_delta

                for hidden_index, hidden_value in enumerate(hidden_outputs):
                    hidden_delta = (
                        previous_output_weights[hidden_index]
                        * output_delta
                        * hidden_value
                        * (1 - hidden_value)
                    )
                    for input_index, input_value in enumerate(inputs):
                        self.hidden_weights[hidden_index][input_index] -= (
                            self.learning_rate * hidden_delta * input_value
                        )
                    self.hidden_biases[hidden_index] -= self.learning_rate * hidden_delta

    def predict(self, inputs):
        _, probability = self.forward(inputs)
        label = 1 if probability >= 0.5 else 0
        return label, probability


def write_results(path, dataset, model):
    correct_predictions = 0
    rows_to_write = []

    for inputs, expected in dataset:
        predicted, probability = model.predict(inputs)
        is_correct = predicted == expected
        correct_predictions += int(is_correct)
        rows_to_write.append(
            {
                "x1": inputs[0],
                "x2": inputs[1],
                "true_label": expected,
                "predicted_label": predicted,
                "probability": f"{probability:.6f}",
                "correct": "yes" if is_correct else "no",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "x1",
                "x2",
                "true_label",
                "predicted_label",
                "probability",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_to_write)

    accuracy = correct_predictions / len(dataset)
    return accuracy


def main():
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / INPUT_FILE
    output_path = base_dir / OUTPUT_FILE

    dataset = read_rows(input_path)
    model = SimpleMLP()
    model.train(dataset)
    accuracy = write_results(output_path, dataset, model)

    print(f"Обработка завершена. Точность на наборе данных: {accuracy:.2%}.")
    print(f"Результаты сохранены в {output_path.name}.")


if __name__ == "__main__":
    main()
