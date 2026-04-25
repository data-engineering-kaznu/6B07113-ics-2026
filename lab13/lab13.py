from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = load_digits()
    images = dataset.images
    features = dataset.data / 16.0
    labels = dataset.target
    return features, labels, images


def save_digit_samples(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    for axis, image, label in zip(axes.flat, images[:10], labels[:10]):
        axis.imshow(image, cmap="gray_r")
        axis.set_title(f"Digit: {label}")
        axis.axis("off")

    fig.suptitle("Examples from the digits dataset", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("MLP confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_loss_curve(loss_curve: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_curve, color="#d97706", linewidth=2)
    ax.set_title("Training loss curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    features, labels, images = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=300,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)

    sample_predictions = model.predict(features[:10])
    sample_image_path = OUTPUT_DIR / "digit_samples.png"
    confusion_path = OUTPUT_DIR / "confusion_matrix.png"
    loss_curve_path = OUTPUT_DIR / "loss_curve.png"

    save_digit_samples(images, labels, sample_image_path)
    save_confusion_matrix(y_test, predictions, confusion_path)
    save_loss_curve(model.named_steps["mlp"].loss_curve_, loss_curve_path)

    print("Lab 13. Artificial neural networks")
    print(f"Dataset size: {len(features)} images")
    print(f"Train size: {len(x_train)}, test size: {len(x_test)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Sample predictions for the first 10 images:")
    print("Actual:   " + " ".join(map(str, labels[:10])))
    print("Predicted:" + " ".join(map(str, sample_predictions)))
    print("\nClassification report:")
    print(report)
    print(f"Examples image saved to: {sample_image_path.name}")
    print(f"Confusion matrix saved to: {confusion_path.name}")
    print(f"Loss curve saved to: {loss_curve_path.name}")


if __name__ == "__main__":
    main()
