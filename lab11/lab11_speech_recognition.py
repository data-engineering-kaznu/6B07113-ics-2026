from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "speech_commands_mfcc.csv"


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_name: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Speech Command Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def plot_command_distribution(labels: pd.Series, output_name: str):
    counts = labels.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"])
    ax.set_title("Speech Command Dataset Distribution")
    ax.set_xlabel("Command")
    ax.set_ylabel("Number of samples")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{int(height)}", ha="center")

    fig.tight_layout()
    fig.savefig(BASE_DIR / output_name, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)
    feature_columns = [column for column in df.columns if column.startswith("mfcc_")]

    X = df[feature_columns]
    y = df["command"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded,
    )

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    class_names = encoder.classes_.tolist()
    cm = confusion_matrix(y_test, y_pred)

    print("Lab 11: Speech Recognition")
    print(f"Dataset rows: {len(df)}")
    print(f"Commands: {class_names}")
    print(f"MFCC feature count: {len(feature_columns)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print()

    test_results = pd.DataFrame(
        {
            "true_label": encoder.inverse_transform(y_test),
            "predicted_label": encoder.inverse_transform(y_pred),
        }
    )
    print("Predictions on test set:")
    print(test_results.to_string(index=False))
    print()

    confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("Confusion matrix:")
    print(confusion_df)

    plot_confusion_matrix(cm, class_names, "confusion_matrix.png")
    plot_command_distribution(df["command"], "command_distribution.png")


if __name__ == "__main__":
    main()
