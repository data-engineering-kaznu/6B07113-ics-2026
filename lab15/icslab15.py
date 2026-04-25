import csv
import math
import random
from pathlib import Path


INPUT_FILE = "input15.csv"
OUTPUT_FILE = "output15.csv"
PIXEL_FIELDS = [f"p{i}" for i in range(16)]
REQUIRED_FIELDS = set(PIXEL_FIELDS + ["label"])
KERNELS = (
    ((1.0, -1.0), (1.0, -1.0)),
    ((1.0, 1.0), (-1.0, -1.0)),
)


def relu(value):
    return value if value > 0 else 0.0


def softmax(values):
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def read_dataset(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            raise ValueError(
                f"Файл {path.name} должен содержать пиксели p0..p15 и столбец label."
            )

        dataset = []
        labels = set()
        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue

            try:
                pixels = [float(row[field]) for field in PIXEL_FIELDS]
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Некорректные данные в строке {line_number}: {row}"
                ) from error

            if label not in (0, 1):
                raise ValueError(f"Метка в строке {line_number} должна быть 0 или 1.")
            if any(pixel < 0 or pixel > 1 for pixel in pixels):
                raise ValueError(
                    f"Пиксели в строке {line_number} должны быть в диапазоне от 0 до 1."
                )

            dataset.append((pixels, label))
            labels.add(label)

        if not dataset:
            raise ValueError(f"Файл {path.name} не содержит данных для обработки.")
        if labels != {0, 1}:
            raise ValueError("Для обучения нужны примеры обоих классов: 0 и 1.")

        return dataset


def pixels_to_image(pixels):
    return [pixels[index:index + 4] for index in range(0, 16, 4)]


def extract_features(pixels):
    image = pixels_to_image(pixels)
    features = []

    for kernel in KERNELS:
        responses = []
        for row in range(3):
            for col in range(3):
                value = 0.0
                for k_row in range(2):
                    for k_col in range(2):
                        value += image[row + k_row][col + k_col] * kernel[k_row][k_col]
                responses.append(relu(value))
        features.append(max(responses))

    return features


class SmallCNNClassifier:
    def __init__(self, feature_count=2, class_count=2, learning_rate=0.2, seed=42):
        random.seed(seed)
        self.learning_rate = learning_rate
        self.weights = [
            [random.uniform(-0.5, 0.5) for _ in range(feature_count)]
            for _ in range(class_count)
        ]
        self.biases = [0.0 for _ in range(class_count)]

    def predict_proba(self, features):
        logits = []
        for class_index in range(len(self.weights)):
            logits.append(
                sum(
                    weight * feature
                    for weight, feature in zip(self.weights[class_index], features)
                )
                + self.biases[class_index]
            )
        return softmax(logits)

    def train(self, feature_rows, epochs=400):
        for _ in range(epochs):
            for features, label in feature_rows:
                probabilities = self.predict_proba(features)
                for class_index in range(len(self.weights)):
                    target = 1.0 if class_index == label else 0.0
                    error = probabilities[class_index] - target
                    for feature_index, feature in enumerate(features):
                        self.weights[class_index][feature_index] -= (
                            self.learning_rate * error * feature
                        )
                    self.biases[class_index] -= self.learning_rate * error

    def predict(self, features):
        probabilities = self.predict_proba(features)
        label = 0 if probabilities[0] >= probabilities[1] else 1
        return label, probabilities


def write_results(path, dataset, model):
    correct = 0
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample",
                "feature_vertical",
                "feature_horizontal",
                "true_label",
                "predicted_label",
                "probability_class_0",
                "probability_class_1",
                "correct",
            ]
        )

        for index, (pixels, label) in enumerate(dataset, start=1):
            features = extract_features(pixels)
            predicted, probabilities = model.predict(features)
            is_correct = predicted == label
            correct += int(is_correct)
            writer.writerow(
                [
                    index,
                    f"{features[0]:.6f}",
                    f"{features[1]:.6f}",
                    label,
                    predicted,
                    f"{probabilities[0]:.6f}",
                    f"{probabilities[1]:.6f}",
                    "yes" if is_correct else "no",
                ]
            )

        writer.writerow([])
        writer.writerow(["accuracy", f"{correct / len(dataset):.2%}"])


def main():
    base_dir = Path(__file__).resolve().parent
    dataset = read_dataset(base_dir / INPUT_FILE)
    feature_rows = [(extract_features(pixels), label) for pixels, label in dataset]

    model = SmallCNNClassifier()
    model.train(feature_rows)
    write_results(base_dir / OUTPUT_FILE, dataset, model)

    print("Обработка завершена. Результаты сохранены в output15.csv.")


if __name__ == "__main__":
    main()
