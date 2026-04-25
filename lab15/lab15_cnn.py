from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import random
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42
IMAGE_SIZE = 32
NUM_CLASSES = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def try_load_cifar10():
    try:
        from tensorflow.keras.datasets import cifar10  # type: ignore
        from tensorflow.keras.models import Sequential  # type: ignore
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
        from tensorflow.keras.utils import to_categorical  # type: ignore
    except Exception:
        return None

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return {
        "mode": "tensorflow",
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "model": model,
        "to_categorical": to_categorical,
    }


def _draw_disk(image: np.ndarray, center: tuple[int, int], radius: int, color: np.ndarray) -> None:
    yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
    image[mask] = np.clip(image[mask] + color, 0.0, 1.0)


def _draw_line(image: np.ndarray, orientation: str, color: np.ndarray, thickness: int = 2) -> None:
    if orientation == "vertical":
        x = IMAGE_SIZE // 2
        image[:, max(0, x - thickness) : min(IMAGE_SIZE, x + thickness + 1)] = np.clip(
            image[:, max(0, x - thickness) : min(IMAGE_SIZE, x + thickness + 1)] + color, 0.0, 1.0
        )
    elif orientation == "horizontal":
        y = IMAGE_SIZE // 2
        image[max(0, y - thickness) : min(IMAGE_SIZE, y + thickness + 1), :] = np.clip(
            image[max(0, y - thickness) : min(IMAGE_SIZE, y + thickness + 1), :] + color, 0.0, 1.0
        )
    elif orientation == "diag":
        for i in range(IMAGE_SIZE):
            for t in range(-thickness, thickness + 1):
                j = i + t
                if 0 <= j < IMAGE_SIZE:
                    image[i, j] = np.clip(image[i, j] + color, 0.0, 1.0)
    elif orientation == "anti_diag":
        for i in range(IMAGE_SIZE):
            j = IMAGE_SIZE - 1 - i
            for t in range(-thickness, thickness + 1):
                jj = j + t
                if 0 <= jj < IMAGE_SIZE:
                    image[i, jj] = np.clip(image[i, jj] + color, 0.0, 1.0)


def _draw_border(image: np.ndarray, color: np.ndarray, thickness: int = 2) -> None:
    image[:thickness, :] = np.clip(image[:thickness, :] + color, 0.0, 1.0)
    image[-thickness:, :] = np.clip(image[-thickness:, :] + color, 0.0, 1.0)
    image[:, :thickness] = np.clip(image[:, :thickness] + color, 0.0, 1.0)
    image[:, -thickness:] = np.clip(image[:, -thickness:] + color, 0.0, 1.0)


def _draw_checker(image: np.ndarray, color_a: np.ndarray, color_b: np.ndarray, block: int = 4) -> None:
    for y in range(0, IMAGE_SIZE, block):
        for x in range(0, IMAGE_SIZE, block):
            color = color_a if ((x // block + y // block) % 2 == 0) else color_b
            image[y : y + block, x : x + block] = np.clip(image[y : y + block, x : x + block] + color, 0.0, 1.0)


def generate_synthetic_dataset(
    samples_per_class_train: int = 60,
    samples_per_class_test: int = 15,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)

    class_names = [
        "circle",
        "vertical",
        "horizontal",
        "diagonal",
        "cross",
        "checker",
        "border",
        "spot_tl_br",
        "spot_tr_bl",
        "plus",
    ]

    colors = [
        np.array([1.0, 0.2, 0.2]),
        np.array([0.2, 1.0, 0.2]),
        np.array([0.2, 0.4, 1.0]),
        np.array([1.0, 0.9, 0.2]),
        np.array([1.0, 0.2, 1.0]),
        np.array([0.2, 1.0, 1.0]),
        np.array([1.0, 0.6, 0.2]),
        np.array([0.9, 0.9, 0.9]),
        np.array([0.7, 0.3, 1.0]),
        np.array([0.8, 1.0, 0.4]),
    ]

    def build_one(label: int) -> np.ndarray:
        image = rng.random((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32) * 0.08
        color = colors[label]
        shift_x = int(rng.integers(-3, 4))
        shift_y = int(rng.integers(-3, 4))

        def shift_center(cx: int, cy: int) -> tuple[int, int]:
            return int(np.clip(cx + shift_x, 2, IMAGE_SIZE - 3)), int(np.clip(cy + shift_y, 2, IMAGE_SIZE - 3))

        if label == 0:
            _draw_disk(image, shift_center(16, 16), 7, color)
        elif label == 1:
            _draw_line(image, "vertical", color, thickness=2)
        elif label == 2:
            _draw_line(image, "horizontal", color, thickness=2)
        elif label == 3:
            _draw_line(image, "diag", color, thickness=2)
        elif label == 4:
            _draw_line(image, "diag", color, thickness=1)
            _draw_line(image, "anti_diag", color, thickness=1)
        elif label == 5:
            _draw_checker(image, color * 0.8, np.array([0.05, 0.05, 0.05]), block=4)
        elif label == 6:
            _draw_border(image, color, thickness=3)
        elif label == 7:
            _draw_disk(image, shift_center(10, 10), 5, color)
            _draw_disk(image, shift_center(22, 22), 5, color * 0.8)
        elif label == 8:
            _draw_disk(image, shift_center(22, 10), 5, color)
            _draw_disk(image, shift_center(10, 22), 5, color * 0.8)
        else:
            _draw_line(image, "vertical", color, thickness=2)
            _draw_line(image, "horizontal", color, thickness=2)

        noise = rng.normal(0.0, 0.04, size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)
        return image

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for label in range(NUM_CLASSES):
        for _ in range(samples_per_class_train):
            x_train.append(build_one(label))
            y_train.append(label)
        for _ in range(samples_per_class_test):
            x_test.append(build_one(label))
            y_test.append(label)

    x_train = np.stack(x_train).astype(np.float32)
    x_test = np.stack(x_test).astype(np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    perm_train = rng.permutation(len(x_train))
    perm_test = rng.permutation(len(x_test))
    x_train, y_train = x_train[perm_train], y_train[perm_train]
    x_test, y_test = x_test[perm_test], y_test[perm_test]

    return {
        "mode": "synthetic",
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "class_names": class_names,
    }


def grayscale(images: np.ndarray) -> np.ndarray:
    return images @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def conv2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i : i + kh, j : j + kw]
            out[i, j] = float(np.sum(region * kernel))
    return out


def max_pool2d(image: np.ndarray, pool_size: int = 2) -> np.ndarray:
    h, w = image.shape
    out_h, out_w = h // pool_size, w // pool_size
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            region = image[i * pool_size : (i + 1) * pool_size, j * pool_size : (j + 1) * pool_size]
            out[i, j] = float(np.max(region))
    return out


FILTERS = np.array(
    [
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
        [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        [[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
    ],
    dtype=np.float32,
)


def extract_cnn_features(images: np.ndarray) -> np.ndarray:
    gray = grayscale(images)
    feature_maps = []
    for image in gray:
        pooled_maps = []
        for kernel in FILTERS:
            conv = conv2d_same(image, kernel)
            activated = np.maximum(conv, 0.0)
            pooled = max_pool2d(activated, pool_size=2)
            pooled_maps.append(pooled)
        feature_maps.append(np.concatenate([m.reshape(-1) for m in pooled_maps]).astype(np.float32))
    return np.stack(feature_maps)


@dataclass
class TrainingHistory:
    losses: list[float]
    accuracies: list[float]


class SoftmaxClassifier:
    def __init__(self, input_dim: int, num_classes: int, seed: int = RANDOM_SEED) -> None:
        rng = np.random.default_rng(seed)
        self.W = (rng.normal(0.0, 0.01, size=(input_dim, num_classes))).astype(np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 60, lr: float = 0.08, batch_size: int = 64) -> TrainingHistory:
        n = len(X)
        y_onehot = np.eye(NUM_CLASSES, dtype=np.float32)[y]
        losses: list[float] = []
        accuracies: list[float] = []

        for _ in range(epochs):
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            for start in range(0, n, batch_size):
                end = start + batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]
                probs = self.predict_proba(xb)
                grad_logits = (probs - yb) / len(xb)
                grad_W = xb.T @ grad_logits
                grad_b = np.sum(grad_logits, axis=0)
                self.W -= lr * grad_W
                self.b -= lr * grad_b

            probs_all = self.predict_proba(X)
            loss = float(-np.mean(np.sum(y_onehot * np.log(probs_all + 1e-9), axis=1)))
            acc = float(np.mean(np.argmax(probs_all, axis=1) == y))
            losses.append(loss)
            accuracies.append(acc)

        return TrainingHistory(losses=losses, accuracies=accuracies)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_history(history: TrainingHistory, output_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history.losses, color="#4c78a8", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.accuracies, color="#e45756", linewidth=2)
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def save_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_name: str, labels: Optional[list[str]] = None) -> None:
    with (BASE_DIR / output_name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true_label", "predicted_label"])
        for t, p in zip(y_true, y_pred):
            if labels:
                writer.writerow([labels[int(t)], labels[int(p)]])
            else:
                writer.writerow([int(t), int(p)])


def save_text(text: str, output_name: str) -> None:
    (BASE_DIR / output_name).write_text(text, encoding="utf-8")


def plot_samples(images: np.ndarray, labels: np.ndarray, output_name: str, class_names: list[str]) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for ax, image, label in zip(axes.flat, images[:10], labels[:10]):
        ax.imshow(image)
        ax.set_title(class_names[int(label)])
        ax.axis("off")
    fig.suptitle("Sample Images", fontsize=14)
    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def run_tensorflow_mode(payload: dict[str, object]) -> None:
    model = payload["model"]
    x_train = payload["x_train"]
    y_train = payload["y_train"]
    x_test = payload["x_test"]
    y_test = payload["y_test"]

    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    class_names = [str(i) for i in range(10)]
    print("Lab 15: Deep Learning and CNN")
    print("Mode: TensorFlow / CIFAR-10")
    print(f"Train samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Test accuracy: {accuracy:.4f}")
    plot_history(TrainingHistory(history.history["loss"], history.history["accuracy"]), "training_history.png")
    plot_confusion_matrix(cm, class_names, "confusion_matrix.png")
    save_predictions(y_test, y_pred, "test_predictions.csv", class_names)


def run_fallback_mode(data: dict[str, np.ndarray]) -> None:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    class_names = data["class_names"]

    train_features = extract_cnn_features(x_train)
    test_features = extract_cnn_features(x_test)

    scaler_mean = train_features.mean(axis=0, keepdims=True)
    scaler_std = train_features.std(axis=0, keepdims=True) + 1e-6
    train_features = (train_features - scaler_mean) / scaler_std
    test_features = (test_features - scaler_mean) / scaler_std

    classifier = SoftmaxClassifier(train_features.shape[1], NUM_CLASSES)
    history = classifier.fit(train_features, y_train, epochs=40, lr=0.09, batch_size=96)

    y_pred_train = classifier.predict(train_features)
    y_pred_test = classifier.predict(test_features)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test, NUM_CLASSES)

    print("Lab 15: Deep Learning and CNN")
    print("Mode: Self-contained CNN-like fallback")
    print("Reason: TensorFlow/Keras is not installed in the local environment.")
    print(f"Train samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Feature dimension: {train_features.shape[1]}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    plot_samples(x_test, y_test, "sample_images.png", class_names)
    plot_history(history, "training_history.png")
    plot_confusion_matrix(cm, class_names, "confusion_matrix.png")
    save_predictions(y_test, y_pred_test, "test_predictions.csv", class_names)
    save_text(
        "\n".join(
            [
                f"mode=fallback",
                f"train_accuracy={train_acc:.4f}",
                f"test_accuracy={test_acc:.4f}",
                f"feature_dimension={train_features.shape[1]}",
            ]
        ),
        "summary.txt",
    )


def main() -> None:
    set_seed(RANDOM_SEED)
    tf_payload = try_load_cifar10()
    if tf_payload is not None:
        run_tensorflow_mode(tf_payload)
        return

    data = generate_synthetic_dataset()
    run_fallback_mode(data)


if __name__ == "__main__":
    main()
