import numpy as np

from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

print("Классификация (Iris)")
# Загрузка данных
iris = load_iris()
X_clf = iris.data
y_clf = iris.target

# пропуски
rng = np.random.default_rng(0)
mask = rng.random(X_clf.shape) < 0.05
X_clf[mask] = np.nan

# Предобработка заполнение пропусков
imputer = SimpleImputer(strategy="mean")
X_clf = imputer.fit_transform(X_clf)

# Нормализация
scaler = StandardScaler()
X_clf = scaler.fit_transform(X_clf)

# Деление
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Модель Логистическая регрессия
model_clf = LogisticRegression(max_iter=200)
model_clf.fit(X_train_clf, y_train_clf)

# Предсказание
y_pred_clf = model_clf.predict(X_test_clf)

# Метрики
accuracy = accuracy_score(y_test_clf, y_pred_clf)
cm = confusion_matrix(y_test_clf, y_pred_clf)

print("Accuracy:", accuracy)
print("Confusion matrix:\n", cm)

# 2. Регрессия (California Housing)
print("\nРегрессия (California Housing)")

# Загрузка данных
housing = fetch_california_housing()
X_reg = housing.data
y_reg = housing.target

# пропуски
rng = np.random.default_rng(1)
mask = rng.random(X_reg.shape) < 0.05
X_reg[mask] = np.nan

# Предобработка заполнение пропусков
imputer = SimpleImputer(strategy="mean")
X_reg = imputer.fit_transform(X_reg)

# Нормализация
scaler = StandardScaler()
X_reg = scaler.fit_transform(X_reg)

# Деление 
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Модель Линейная регрессия
model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)

# Предсказание
y_pred_reg = model_reg.predict(X_test_reg)

# Метрики
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("MSE:", mse)
print("R2:", r2)
