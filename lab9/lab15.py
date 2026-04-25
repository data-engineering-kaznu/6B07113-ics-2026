import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# ── Загрузка и подготовка данных ───────────────────────────────────────────────

print("Загрузка CIFAR-10...")
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Нормализация пикселей: 0-255 → 0-1
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

# One-hot кодирование меток
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test,  10)

print(f"Обучающая выборка: {X_train.shape}")  # (50000, 32, 32, 3)
print(f"Тестовая выборка:  {X_test.shape}")   # (10000, 32, 32, 3)


# ── Архитектура CNN ────────────────────────────────────────────────────────────

model = keras.Sequential([
    # Первый блок свёртки: находит простые паттерны (края, текстуры)
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Второй блок свёртки: находит сложные паттерны (формы, части объектов)
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Классификатор
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.summary()


# ── Компиляция и обучение ──────────────────────────────────────────────────────

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)


# ── Итог ───────────────────────────────────────────────────────────────────────

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nФинальная точность на тестовых данных: {accuracy * 100:.2f}%")
if accuracy >= 0.70:
    print("Результат: цель достигнута (>= 70%)")
else:
    print("Результат: цель не достигнута (< 70%)")
