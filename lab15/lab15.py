from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report


IMAGE_SIZE = 32
N_CLASSES = 10
TRAIN_SAMPLES_PER_CLASS = 120
TEST_SAMPLES_PER_CLASS = 30
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.01
OUTPUT_DIR = Path(__file__).resolve().parent
CLASS_NAMES = (
    "vertical_red",
    "horizontal_green",
    "diag_blue",
    "anti_diag_yellow",
    "square_magenta",
    "circle_cyan",
    "triangle_orange",
    "x_white",
    "plus_teal",
    "checker_mix",
)


@dataclass
class TinyCNN:
    filters: np.ndarray
    bias: np.ndarray
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    dense_weights: np.ndarray
    dense_bias: np.ndarray


def draw_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: np.ndarray, thickness: int) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.linspace(x0, x1, steps).astype(int)
    ys = np.linspace(y0, y1, steps).astype(int)

    for x, y in zip(xs, ys):
        x_min = max(0, x - thickness)
        x_max = min(IMAGE_SIZE, x + thickness + 1)
        y_min = max(0, y - thickness)
        y_max = min(IMAGE_SIZE, y + thickness + 1)
        image[y_min:y_max, x_min:x_max] = color


def draw_rectangle(image: np.ndarray, top: int, left: int, size: int, color: np.ndarray, thickness: int) -> None:
    bottom = min(IMAGE_SIZE - 1, top + size)
    right = min(IMAGE_SIZE - 1, left + size)
    image[top : top + thickness, left:right] = color
    image[bottom - thickness : bottom, left:right] = color
    image[top:bottom, left : left + thickness] = color
    image[top:bottom, right - thickness : right] = color


def draw_circle(image: np.ndarray, center_x: int, center_y: int, radius: int, color: np.ndarray, thickness: int) -> None:
    yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    mask = (distance >= radius - thickness) & (distance <= radius + thickness)
    image[mask] = color


def draw_triangle(image: np.ndarray, color: np.ndarray, thickness: int) -> None:
    draw_line(image, (16, 5), (6, 25), color, thickness)
    draw_line(image, (16, 5), (26, 25), color, thickness)
    draw_line(image, (6, 25), (26, 25), color, thickness)


def generate_pattern(label: int, rng: np.random.Generator) -> np.ndarray:
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 0.08, dtype=np.float32)
    image += rng.normal(0.0, 0.015, size=image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    shift_x = int(rng.integers(-2, 3))
    shift_y = int(rng.integers(-2, 3))
    thickness = int(rng.integers(1, 3))

    red = np.array([0.92, 0.18, 0.18], dtype=np.float32)
    green = np.array([0.18, 0.88, 0.28], dtype=np.float32)
    blue = np.array([0.20, 0.42, 0.95], dtype=np.float32)
    yellow = np.array([0.95, 0.88, 0.18], dtype=np.float32)
    magenta = np.array([0.86, 0.22, 0.82], dtype=np.float32)
    cyan = np.array([0.18, 0.92, 0.90], dtype=np.float32)
    orange = np.array([0.96, 0.52, 0.12], dtype=np.float32)
    white = np.array([0.94, 0.94, 0.94], dtype=np.float32)
    teal = np.array([0.15, 0.70, 0.70], dtype=np.float32)

    if label == 0:
        x = 16 + shift_x
        image[:, max(0, x - 2) : min(IMAGE_SIZE, x + 2)] = red
    elif label == 1:
        y = 16 + shift_y
        image[max(0, y - 2) : min(IMAGE_SIZE, y + 2), :] = green
    elif label == 2:
        draw_line(image, (4 + shift_x, 4 + shift_y), (27 + shift_x, 27 + shift_y), blue, thickness)
    elif label == 3:
        draw_line(image, (27 + shift_x, 4 + shift_y), (4 + shift_x, 27 + shift_y), yellow, thickness)
    elif label == 4:
        draw_rectangle(image, 7 + shift_y, 7 + shift_x, 18, magenta, thickness + 1)
    elif label == 5:
        draw_circle(image, 16 + shift_x, 16 + shift_y, 8, cyan, thickness + 1)
    elif label == 6:
        temp = np.zeros_like(image)
        draw_triangle(temp, orange, thickness)
        image = np.maximum(image, np.roll(np.roll(temp, shift_y, axis=0), shift_x, axis=1))
    elif label == 7:
        draw_line(image, (5 + shift_x, 5 + shift_y), (26 + shift_x, 26 + shift_y), white, thickness)
        draw_line(image, (26 + shift_x, 5 + shift_y), (5 + shift_x, 26 + shift_y), white, thickness)
    elif label == 8:
        x = 16 + shift_x
        y = 16 + shift_y
        image[:, max(0, x - 1) : min(IMAGE_SIZE, x + 2)] = teal
        image[max(0, y - 1) : min(IMAGE_SIZE, y + 2), :] = teal
    else:
        tile = 4 + int(rng.integers(0, 2))
        colors = np.array(
            [
                [0.92, 0.20, 0.20],
                [0.20, 0.75, 0.92],
                [0.96, 0.86, 0.25],
            ],
            dtype=np.float32,
        )
        for row in range(0, IMAGE_SIZE, tile):
            for col in range(0, IMAGE_SIZE, tile):
                color = colors[(row // tile + col // tile) % len(colors)]
                image[row : row + tile, col : col + tile] = color

    image += rng.normal(0.0, 0.03, size=image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def build_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x_train: list[np.ndarray] = []
    y_train: list[int] = []
    x_test: list[np.ndarray] = []
    y_test: list[int] = []

    for label in range(N_CLASSES):
        for _ in range(TRAIN_SAMPLES_PER_CLASS):
            x_train.append(generate_pattern(label, rng))
            y_train.append(label)
        for _ in range(TEST_SAMPLES_PER_CLASS):
            x_test.append(generate_pattern(label, rng))
            y_test.append(label)

    x_train_array = np.array(x_train, dtype=np.float32)
    y_train_array = np.array(y_train, dtype=np.int64)
    x_test_array = np.array(x_test, dtype=np.float32)
    y_test_array = np.array(y_test, dtype=np.int64)

    train_perm = rng.permutation(len(x_train_array))
    test_perm = rng.permutation(len(x_test_array))
    return (
        x_train_array[train_perm],
        y_train_array[train_perm],
        x_test_array[test_perm],
        y_test_array[test_perm],
    )


def initialize_model(random_state: int = 42) -> TinyCNN:
    rng = np.random.default_rng(random_state)
    filters = rng.normal(0.0, 0.12, size=(8, 3, 3, 3)).astype(np.float32)
    bias = np.zeros(8, dtype=np.float32)
    hidden_weights = rng.normal(0.0, 0.08, size=(15 * 15 * 8, 64)).astype(np.float32)
    hidden_bias = np.zeros(64, dtype=np.float32)
    dense_weights = rng.normal(0.0, 0.08, size=(64, N_CLASSES)).astype(np.float32)
    dense_bias = np.zeros(N_CLASSES, dtype=np.float32)
    return TinyCNN(filters, bias, hidden_weights, hidden_bias, dense_weights, dense_bias)


def conv_forward(x: np.ndarray, filters: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, height, width, channels = x.shape
    n_filters, kernel_h, kernel_w, _ = filters.shape
    out_h = height - kernel_h + 1
    out_w = width - kernel_w + 1
    output = np.zeros((batch, out_h, out_w, n_filters), dtype=np.float32)

    for row in range(out_h):
        for col in range(out_w):
            patch = x[:, row : row + kernel_h, col : col + kernel_w, :]
            output[:, row, col, :] = np.tensordot(
                patch,
                filters,
                axes=([1, 2, 3], [1, 2, 3]),
            ) + bias

    return output


def maxpool_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch, height, width, channels = x.shape
    x_reshaped = x.reshape(batch, height // 2, 2, width // 2, 2, channels)
    pooled = x_reshaped.max(axis=(2, 4))
    mask = x_reshaped == pooled[:, :, None, :, None, :]
    return pooled, mask


def maxpool_backward(grad_output: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grad = grad_output[:, :, None, :, None, :] * mask
    normalizer = np.maximum(mask.sum(axis=(2, 4), keepdims=True), 1)
    grad = grad / normalizer
    batch, pooled_h, _, pooled_w, _, channels = grad.shape
    return grad.reshape(batch, pooled_h * 2, pooled_w * 2, channels)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / np.sum(exponents, axis=1, keepdims=True)


def forward_pass(model: TinyCNN, x: np.ndarray) -> dict[str, np.ndarray]:
    conv = conv_forward(x, model.filters, model.bias)
    relu = np.maximum(conv, 0.0)
    pooled, mask = maxpool_forward(relu)
    flattened = pooled.reshape(len(x), -1)
    hidden_pre = flattened @ model.hidden_weights + model.hidden_bias
    hidden = np.maximum(hidden_pre, 0.0)
    logits = hidden @ model.dense_weights + model.dense_bias
    probabilities = softmax(logits)
    return {
        "conv": conv,
        "relu": relu,
        "pooled": pooled,
        "pool_mask": mask,
        "flattened": flattened,
        "hidden_pre": hidden_pre,
        "hidden": hidden,
        "logits": logits,
        "probabilities": probabilities,
    }


def backward_pass(model: TinyCNN, x: np.ndarray, y: np.ndarray, cache: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    batch_size = x.shape[0]
    probabilities = cache["probabilities"].copy()
    probabilities[np.arange(batch_size), y] -= 1.0
    probabilities /= batch_size

    grad_dense_weights = cache["hidden"].T @ probabilities
    grad_dense_bias = probabilities.sum(axis=0)

    grad_hidden = probabilities @ model.dense_weights.T
    grad_hidden_pre = grad_hidden * (cache["hidden_pre"] > 0.0)
    grad_hidden_weights = cache["flattened"].T @ grad_hidden_pre
    grad_hidden_bias = grad_hidden_pre.sum(axis=0)

    grad_flattened = grad_hidden_pre @ model.hidden_weights.T
    pooled = cache["pooled"]
    grad_pooled = grad_flattened.reshape(pooled.shape)
    grad_relu = maxpool_backward(grad_pooled, cache["pool_mask"])
    grad_conv = grad_relu * (cache["conv"] > 0.0)

    grad_filters = np.zeros_like(model.filters)
    grad_bias = grad_conv.sum(axis=(0, 1, 2))
    grad_input = np.zeros_like(x)

    kernel_h = model.filters.shape[1]
    kernel_w = model.filters.shape[2]
    out_h = grad_conv.shape[1]
    out_w = grad_conv.shape[2]

    for row in range(out_h):
        for col in range(out_w):
            patch = x[:, row : row + kernel_h, col : col + kernel_w, :]
            grad_filters += np.einsum("bf,bhwc->fhwc", grad_conv[:, row, col, :], patch)
            grad_input[:, row : row + kernel_h, col : col + kernel_w, :] += np.einsum(
                "bf,fhwc->bhwc",
                grad_conv[:, row, col, :],
                model.filters,
            )

    gradients = {
        "filters": grad_filters,
        "bias": grad_bias,
        "hidden_weights": grad_hidden_weights,
        "hidden_bias": grad_hidden_bias,
        "dense_weights": grad_dense_weights,
        "dense_bias": grad_dense_bias,
    }

    true_probabilities = np.clip(cache["probabilities"][np.arange(batch_size), y], 1e-8, 1.0)
    loss = float(-np.mean(np.log(true_probabilities)))
    return gradients, loss


def update_model(model: TinyCNN, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
    model.filters -= learning_rate * gradients["filters"]
    model.bias -= learning_rate * gradients["bias"]
    model.hidden_weights -= learning_rate * gradients["hidden_weights"]
    model.hidden_bias -= learning_rate * gradients["hidden_bias"]
    model.dense_weights -= learning_rate * gradients["dense_weights"]
    model.dense_bias -= learning_rate * gradients["dense_bias"]


def predict(model: TinyCNN, x: np.ndarray) -> np.ndarray:
    cache = forward_pass(model, x)
    return np.argmax(cache["probabilities"], axis=1)


def train_model(model: TinyCNN, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(42)
    loss_history: list[float] = []
    accuracy_history: list[float] = []

    for epoch in range(EPOCHS):
        permutation = rng.permutation(len(x_train))
        x_shuffled = x_train[permutation]
        y_shuffled = y_train[permutation]
        batch_losses: list[float] = []

        for start in range(0, len(x_shuffled), BATCH_SIZE):
            stop = start + BATCH_SIZE
            batch_x = x_shuffled[start:stop]
            batch_y = y_shuffled[start:stop]

            cache = forward_pass(model, batch_x)
            gradients, loss = backward_pass(model, batch_x, batch_y, cache)
            update_model(model, gradients, LEARNING_RATE)
            batch_losses.append(loss)

        test_predictions = predict(model, x_test)
        accuracy = accuracy_score(y_test, test_predictions)
        loss_history.append(float(np.mean(batch_losses)))
        accuracy_history.append(float(accuracy))
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} - "
            f"loss: {loss_history[-1]:.4f} - test_accuracy: {accuracy_history[-1]:.4f}"
        )

    return loss_history, accuracy_history


def save_sample_grid(images: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for class_index, axis in enumerate(axes.flat):
        sample = images[np.where(labels == class_index)[0][0]]
        axis.imshow(sample)
        axis.set_title(CLASS_NAMES[class_index], fontsize=8)
        axis.axis("off")

    figure.suptitle("Synthetic CIFAR-like dataset samples", fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_training_curves(loss_history: list[float], accuracy_history: list[float], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(loss_history) + 1)

    axes[0].plot(epochs, loss_history, color="#dc2626", linewidth=2)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, accuracy_history, color="#2563eb", linewidth=2)
    axes[1].set_title("Test accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Tiny CNN confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    x_train, y_train, x_test, y_test = build_dataset()
    model = initialize_model()
    loss_history, accuracy_history = train_model(model, x_train, y_train, x_test, y_test)

    predictions = predict(model, x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=CLASS_NAMES, digits=4)

    samples_path = OUTPUT_DIR / "dataset_samples.png"
    curves_path = OUTPUT_DIR / "training_curves.png"
    confusion_path = OUTPUT_DIR / "confusion_matrix.png"

    save_sample_grid(x_train, y_train, samples_path)
    save_training_curves(loss_history, accuracy_history, curves_path)
    save_confusion_matrix(y_test, predictions, confusion_path)

    print("Lab 15. Deep learning and convolutional neural networks")
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print(f"Final test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)
    print(f"Dataset samples saved to: {samples_path.name}")
    print(f"Training curves saved to: {curves_path.name}")
    print(f"Confusion matrix saved to: {confusion_path.name}")


if __name__ == "__main__":
    main()
