from __future__ import annotations

import os
from pathlib import Path

import numpy as np

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
SAMPLE_RATE = 16_000
DURATION_SECONDS = 1.0
COMMANDS = ["forward", "back", "stop"]
RANDOM_SEED = 11


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595 * np.log10(1 + np.asarray(frequency) / 700)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700 * (10 ** (np.asarray(mel) / 2595) - 1)


def dct_type_2(values: np.ndarray, coefficient_count: int) -> np.ndarray:
    sample_count = values.shape[0]
    indexes = np.arange(sample_count)
    coefficients = []

    for coefficient_index in range(coefficient_count):
        basis = np.cos(
            np.pi * coefficient_index * (2 * indexes + 1) / (2 * sample_count)
        )
        coefficients.append(np.sum(values * basis))

    return np.array(coefficients)


def create_mel_filter_bank(
    sample_rate: int,
    fft_size: int,
    filter_count: int = 26,
    min_frequency: int = 0,
    max_frequency: int | None = None,
) -> np.ndarray:
    if max_frequency is None:
        max_frequency = sample_rate // 2

    mel_points = np.linspace(
        hz_to_mel(min_frequency),
        hz_to_mel(max_frequency),
        filter_count + 2,
    )
    frequency_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * frequency_points / sample_rate).astype(int)

    filters = np.zeros((filter_count, fft_size // 2 + 1))
    for filter_index in range(1, filter_count + 1):
        left = bin_points[filter_index - 1]
        center = bin_points[filter_index]
        right = bin_points[filter_index + 1]

        for fft_bin in range(left, center):
            filters[filter_index - 1, fft_bin] = (fft_bin - left) / max(center - left, 1)
        for fft_bin in range(center, right):
            filters[filter_index - 1, fft_bin] = (right - fft_bin) / max(right - center, 1)

    return filters


def extract_mfcc(
    signal: np.ndarray,
    sample_rate: int,
    coefficient_count: int = 13,
    frame_length_ms: int = 25,
    frame_step_ms: int = 10,
) -> np.ndarray:
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_step = int(sample_rate * frame_step_ms / 1000)
    frame_count = 1 + int(np.ceil((len(emphasized) - frame_length) / frame_step))
    padded_length = frame_count * frame_step + frame_length
    padded_signal = np.pad(emphasized, (0, padded_length - len(emphasized)))

    indexes = (
        np.arange(frame_length)[None, :]
        + np.arange(frame_count)[:, None] * frame_step
    )
    frames = padded_signal[indexes]
    frames *= np.hamming(frame_length)

    fft_size = 512
    spectrum = np.fft.rfft(frames, fft_size)
    power_spectrum = (1 / fft_size) * np.abs(spectrum) ** 2

    filters = create_mel_filter_bank(sample_rate, fft_size)
    filter_energies = np.dot(power_spectrum, filters.T)
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
    log_energies = np.log(filter_energies)

    mfcc = np.array(
        [dct_type_2(frame, coefficient_count) for frame in log_energies]
    )
    return mfcc


def summarize_mfcc(mfcc: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            mfcc.mean(axis=0),
            mfcc.std(axis=0),
            mfcc.min(axis=0),
            mfcc.max(axis=0),
        ]
    )


def generate_command_signal(command: str, rng: np.random.Generator) -> np.ndarray:
    time = np.linspace(0, DURATION_SECONDS, int(SAMPLE_RATE * DURATION_SECONDS), endpoint=False)
    phase_shift = rng.uniform(-0.15, 0.15)
    amplitude = rng.uniform(0.75, 1.0)

    if command == "forward":
        signal = (
            np.sin(2 * np.pi * (360 + 90 * time) * time + phase_shift)
            + 0.35 * np.sin(2 * np.pi * 720 * time)
        )
    elif command == "back":
        signal = (
            np.sin(2 * np.pi * (780 - 120 * time) * time + phase_shift)
            + 0.30 * np.sin(2 * np.pi * 390 * time)
        )
    elif command == "stop":
        envelope = np.where(time < 0.45, 1.0, 0.4)
        signal = envelope * (
            np.sin(2 * np.pi * 520 * time + phase_shift)
            + 0.45 * np.sin(2 * np.pi * 1040 * time)
        )
    else:
        raise ValueError(f"Unknown command: {command}")

    noise = rng.normal(0, 0.04, size=time.shape)
    signal = amplitude * signal + noise
    signal /= np.max(np.abs(signal))
    return signal


def build_dataset(samples_per_command: int = 45) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(RANDOM_SEED)
    features = []
    labels = []
    example_signals = []

    for label_index, command in enumerate(COMMANDS):
        for sample_index in range(samples_per_command):
            signal = generate_command_signal(command, rng)
            mfcc = extract_mfcc(signal, SAMPLE_RATE)
            features.append(summarize_mfcc(mfcc))
            labels.append(label_index)

            if sample_index == 0:
                example_signals.append(signal)

    return np.array(features), np.array(labels), example_signals


def train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_ratio: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    indexes = rng.permutation(len(features))
    test_size = int(len(features) * test_ratio)
    test_indexes = indexes[:test_size]
    train_indexes = indexes[test_size:]
    return (
        features[train_indexes],
        features[test_indexes],
        labels[train_indexes],
        labels[test_indexes],
    )


def one_hot(labels: np.ndarray, class_count: int) -> np.ndarray:
    encoded = np.zeros((len(labels), class_count))
    encoded[np.arange(len(labels)), labels] = 1
    return encoded


def standardize(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std == 0, 1, std)
    return (train_features - mean) / std, (test_features - mean) / std


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def train_neural_network(
    features: np.ndarray,
    labels: np.ndarray,
    hidden_size: int = 32,
    epochs: int = 450,
    learning_rate: float = 0.04,
) -> tuple[dict[str, np.ndarray], list[float]]:
    rng = np.random.default_rng(RANDOM_SEED)
    sample_count, feature_count = features.shape
    class_count = len(COMMANDS)
    targets = one_hot(labels, class_count)

    weights_1 = rng.normal(0, 0.12, size=(feature_count, hidden_size))
    bias_1 = np.zeros((1, hidden_size))
    weights_2 = rng.normal(0, 0.12, size=(hidden_size, class_count))
    bias_2 = np.zeros((1, class_count))
    losses = []

    for _ in range(epochs):
        hidden_raw = features @ weights_1 + bias_1
        hidden = np.tanh(hidden_raw)
        logits = hidden @ weights_2 + bias_2
        probabilities = softmax(logits)

        loss = -np.mean(np.sum(targets * np.log(probabilities + 1e-12), axis=1))
        losses.append(float(loss))

        output_gradient = (probabilities - targets) / sample_count
        weights_2 -= learning_rate * hidden.T @ output_gradient
        bias_2 -= learning_rate * output_gradient.sum(axis=0, keepdims=True)

        hidden_gradient = (output_gradient @ weights_2.T) * (1 - hidden**2)
        weights_1 -= learning_rate * features.T @ hidden_gradient
        bias_1 -= learning_rate * hidden_gradient.sum(axis=0, keepdims=True)

    model = {
        "weights_1": weights_1,
        "bias_1": bias_1,
        "weights_2": weights_2,
        "bias_2": bias_2,
    }
    return model, losses


def predict(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    hidden = np.tanh(features @ model["weights_1"] + model["bias_1"])
    logits = hidden @ model["weights_2"] + model["bias_2"]
    return np.argmax(softmax(logits), axis=1)


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(COMMANDS), len(COMMANDS)), dtype=int)
    for actual_label, predicted_label in zip(actual, predicted):
        matrix[actual_label, predicted_label] += 1
    return matrix


def plot_waveform(example_signals: list[np.ndarray]) -> None:
    time = np.linspace(0, DURATION_SECONDS, len(example_signals[0]), endpoint=False)
    plt.figure(figsize=(9, 5))
    for command, signal in zip(COMMANDS, example_signals):
        plt.plot(time, signal, label=command, alpha=0.8)
    plt.xlabel("Time, seconds")
    plt.ylabel("Amplitude")
    plt.title("Synthetic Speech Commands")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "waveform.png", dpi=150)
    plt.close()


def plot_mfcc(signal: np.ndarray) -> None:
    mfcc = extract_mfcc(signal, SAMPLE_RATE)
    plt.figure(figsize=(8, 5))
    plt.imshow(mfcc.T, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Coefficient value")
    plt.xlabel("Frame")
    plt.ylabel("MFCC coefficient")
    plt.title("MFCC Features for Command")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mfcc_features.png", dpi=150)
    plt.close()


def plot_training_loss(losses: list[float]) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Neural Network Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray) -> None:
    plt.figure(figsize=(6, 5))
    image = plt.imshow(matrix, cmap="Blues")
    plt.colorbar(image, label="Samples")
    plt.xticks(range(len(COMMANDS)), COMMANDS)
    plt.yticks(range(len(COMMANDS)), COMMANDS)
    plt.xlabel("Predicted command")
    plt.ylabel("Actual command")
    plt.title("Speech Command Confusion Matrix")

    max_value = matrix.max()
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            plt.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > max_value / 2 else "black",
                fontsize=12,
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    features, labels, example_signals = build_dataset()
    train_x, test_x, train_y, test_y = train_test_split(features, labels)
    train_x, test_x = standardize(train_x, test_x)

    model, losses = train_neural_network(train_x, train_y)
    predictions = predict(model, test_x)
    accuracy = float(np.mean(predictions == test_y))
    matrix = confusion_matrix(test_y, predictions)

    print("Lab 11: Speech Recognition System")
    print("-" * 64)
    print("Commands:", ", ".join(COMMANDS))
    print(f"Training samples: {len(train_x)}")
    print(f"Testing samples: {len(test_x)}")
    print(f"Test accuracy: {accuracy:.2%}")
    print()
    print(f"{'Actual':>10} | {'Predicted':>10}")
    print("-" * 25)
    for actual_label, predicted_label in zip(test_y[:10], predictions[:10]):
        print(f"{COMMANDS[actual_label]:>10} | {COMMANDS[predicted_label]:>10}")
    print()
    print(f"Plots saved to: {OUTPUT_DIR}")

    plot_waveform(example_signals)
    plot_mfcc(example_signals[0])
    plot_training_loss(losses)
    plot_confusion_matrix(matrix)


if __name__ == "__main__":
    main()
