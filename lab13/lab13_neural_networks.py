from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent


def plot_sample_digits(images: np.ndarray, labels: np.ndarray, output_name: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Digits Dataset: Sample Images", fontsize=14)

    for ax, image, label in zip(axes.flat, images[:10], labels[:10]):
        ax.imshow(image, cmap="gray_r")
        ax.set_title(f"Digit {label}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix for Digit Classification")
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def plot_loss_curve(loss_curve: list[float], output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_curve) + 1), loss_curve, color="#4c78a8", linewidth=2)
    ax.set_title("MLP Training Loss Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def save_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_name: str) -> None:
    df = pd.DataFrame({"true_digit": y_true, "predicted_digit": y_pred})
    df.to_csv(BASE_DIR / output_name, index=False)


def save_report(report_text: str, output_name: str) -> None:
    (BASE_DIR / output_name).write_text(report_text, encoding="utf-8")


def main() -> None:
    digits = load_digits()
    X = digits.data
    y = digits.target
    images = digits.images

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, train_images, test_images = train_test_split(
        X_scaled,
        y,
        images,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4)

    print("Lab 13: Artificial Neural Networks")
    print(f"Dataset: sklearn digits")
    print(f"Samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Hidden layers: {model.hidden_layer_sizes}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Test accuracy: {accuracy:.4f}")
    print()
    print("Classification report:")
    print(report_text)

    preview_df = pd.DataFrame(
        {
            "true_digit": y_test[:20],
            "predicted_digit": y_pred[:20],
        }
    )
    print("First 20 predictions:")
    print(preview_df.to_string(index=False))

    plot_sample_digits(test_images, y_test, "digits_samples.png")
    plot_confusion_matrix(cm, [str(i) for i in range(10)], "confusion_matrix.png")
    plot_loss_curve(model.loss_curve_, "training_loss.png")
    save_predictions(y_test, y_pred, "test_predictions.csv")
    save_report(report_text, "classification_report.txt")


if __name__ == "__main__":
    main()
