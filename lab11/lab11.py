from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


SAMPLE_RATE = 16_000
COMMANDS = ("forward", "backward", "stop")
OUTPUT_DIR = Path(__file__).resolve().parent


def hz_to_mel(frequency_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency_hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_filters: int = 26,
    min_hz: float = 0.0,
    max_hz: float | None = None,
) -> np.ndarray:
    max_hz = max_hz or sample_rate / 2
    mel_points = np.linspace(hz_to_mel(min_hz), hz_to_mel(max_hz), n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_filters, n_fft // 2 + 1), dtype=float)
    for index in range(1, n_filters + 1):
        left = bins[index - 1]
        center = bins[index]
        right = bins[index + 1]

        if center == left:
            center += 1
        if right == center:
            right += 1

        for freq_bin in range(left, center):
            filterbank[index - 1, freq_bin] = (freq_bin - left) / (center - left)
        for freq_bin in range(center, right):
            filterbank[index - 1, freq_bin] = (right - freq_bin) / (right - center)

    return filterbank


def frame_signal(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if signal.size < frame_size:
        padding = np.zeros(frame_size - signal.size, dtype=float)
        signal = np.concatenate([signal, padding])

    n_frames = 1 + int(np.ceil((signal.size - frame_size) / hop_size))
    padded_length = frame_size + (n_frames - 1) * hop_size
    padded_signal = np.pad(signal, (0, padded_length - signal.size))

    indices = (
        np.arange(frame_size)[None, :] + hop_size * np.arange(n_frames)[:, None]
    )
    return padded_signal[indices]


def compute_mfcc(
    signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = 13,
    n_filters: int = 26,
    frame_ms: float = 0.025,
    hop_ms: float = 0.010,
) -> np.ndarray:
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frame_size = int(sample_rate * frame_ms)
    hop_size = int(sample_rate * hop_ms)
    frames = frame_signal(emphasized, frame_size, hop_size)

    windowed = frames * np.hamming(frame_size)
    n_fft = 512
    power_spectrum = (1.0 / n_fft) * np.square(
        np.abs(np.fft.rfft(windowed, n=n_fft))
    )

    filters = mel_filterbank(sample_rate, n_fft, n_filters=n_filters)
    mel_energy = np.dot(power_spectrum, filters.T)
    mel_energy = np.where(mel_energy == 0.0, np.finfo(float).eps, mel_energy)

    cepstrum = dct(np.log(mel_energy), type=2, axis=1, norm="ortho")[:, :n_mfcc]
    cepstrum -= np.mean(cepstrum, axis=0, keepdims=True)
    return cepstrum


def generate_command_signal(
    command: str,
    rng: np.random.Generator,
    sample_rate: int = SAMPLE_RATE,
    duration: float = 1.0,
) -> np.ndarray:
    time = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    base_frequencies = {
        "forward": (220.0, 440.0, 660.0),
        "backward": (180.0, 360.0, 540.0),
        "stop": (260.0, 520.0, 780.0),
    }[command]

    signal = np.zeros_like(time)
    for index, base_frequency in enumerate(base_frequencies, start=1):
        frequency = base_frequency + rng.normal(0.0, 6.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = 0.45 / index
        signal += amplitude * np.sin(2.0 * np.pi * frequency * time + phase)

    envelope = np.hanning(time.size)
    modulation = 1.0 + 0.15 * np.sin(2.0 * np.pi * (2.0 + rng.random()) * time)
    noise = rng.normal(0.0, 0.02, size=time.size)

    return (signal * envelope * modulation + noise).astype(np.float32)


def build_dataset(
    samples_per_command: int = 60,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    features: list[np.ndarray] = []
    labels: list[str] = []
    sample_mfcc = None

    for command in COMMANDS:
        for sample_index in range(samples_per_command):
            signal = generate_command_signal(command, rng)
            mfcc = compute_mfcc(signal)
            if command == "forward" and sample_index == 0:
                sample_mfcc = mfcc

            summary = np.concatenate(
                [
                    np.mean(mfcc, axis=0),
                    np.std(mfcc, axis=0),
                    np.max(mfcc, axis=0),
                ]
            )
            features.append(summary)
            labels.append(command)

    return np.array(features), np.array(labels), sample_mfcc


def save_mfcc_plot(mfcc: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(mfcc.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("MFCC for sample command 'forward'")
    ax.set_xlabel("Frame")
    ax.set_ylabel("MFCC coefficient")
    fig.colorbar(image, ax=ax, label="Value")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, ...],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Speech command classification")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    features, labels, sample_mfcc = build_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        stratify=labels,
        random_state=42,
    )

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=800,
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    mfcc_plot = OUTPUT_DIR / "mfcc_example.png"
    confusion_plot = OUTPUT_DIR / "confusion_matrix.png"
    save_mfcc_plot(sample_mfcc, mfcc_plot)
    save_confusion_matrix(y_test, predictions, COMMANDS, confusion_plot)

    test_signal = generate_command_signal("stop", np.random.default_rng(7))
    test_mfcc = compute_mfcc(test_signal)
    test_feature = np.concatenate(
        [
            np.mean(test_mfcc, axis=0),
            np.std(test_mfcc, axis=0),
            np.max(test_mfcc, axis=0),
        ]
    ).reshape(1, -1)
    predicted_command = model.predict(test_feature)[0]
    class_probabilities = model.predict_proba(test_feature)[0]

    print("Lab 11. Speech recognition demo")
    print(f"Commands: {', '.join(COMMANDS)}")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=3))
    print("Test command prediction:")
    print(f"Predicted label: {predicted_command}")
    print(
        "Class probabilities:",
        {
            str(label): round(float(probability), 3)
            for label, probability in zip(model.classes_, class_probabilities)
        },
    )
    print(f"MFCC plot saved to: {mfcc_plot.name}")
    print(f"Confusion matrix saved to: {confusion_plot.name}")


if __name__ == "__main__":
    main()
