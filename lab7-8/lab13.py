import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# ── Функции активации ──────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Функция потерь ─────────────────────────────────────────────────────────────

def cross_entropy_loss(y_pred, y_true):
    n = y_true.shape[0]
    log_probs = -np.log(y_pred[range(n), y_true.argmax(axis=1)] + 1e-8)
    return log_probs.mean()


# ── MLP модель ─────────────────────────────────────────────────────────────────

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        # Инициализация весов (метод He для ReLU)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, y_true):
        n = y_true.shape[0]

        # Градиент выходного слоя
        dz2 = self.a2 - y_true
        dW2 = self.a1.T @ dz2 / n
        db2 = dz2.mean(axis=0, keepdims=True)

        # Градиент скрытого слоя
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = self.X.T @ dz1 / n
        db1 = dz1.mean(axis=0, keepdims=True)

        # Обновление весов
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X, y):
        return (self.predict(X) == y.argmax(axis=1)).mean()


# ── Загрузка и подготовка данных ───────────────────────────────────────────────

print("Загрузка MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int)

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=10000, random_state=42
)

def to_categorical(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train = to_categorical(y_train_raw)
y_test  = to_categorical(y_test_raw)


# ── Обучение ───────────────────────────────────────────────────────────────────

model = MLP(input_size=784, hidden_size=128, output_size=10, lr=0.1)

epochs     = 10
batch_size = 256
n          = X_train.shape[0]

for epoch in range(epochs):
    # Перемешиваем данные каждую эпоху
    idx = np.random.permutation(n)
    X_shuffled = X_train[idx]
    y_shuffled = y_train[idx]

    for i in range(0, n, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        y_pred = model.forward(X_batch)
        loss   = cross_entropy_loss(y_pred, y_batch)
        model.backward(y_batch)

    train_acc = model.accuracy(X_train, y_train)
    test_acc  = model.accuracy(X_test,  y_test)
    print(f"Epoch {epoch + 1:2d}/{epochs} | Loss: {loss:.4f} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")


# ── Итог ───────────────────────────────────────────────────────────────────────

final_acc = model.accuracy(X_test, y_test)
print(f"\nФинальная точность на тестовых данных: {final_acc * 100:.2f}%")
if final_acc >= 0.90:
    print("Результат: цель достигнута (>= 90%)")
else:
    print("Результат: цель не достигнута (< 90%)")
