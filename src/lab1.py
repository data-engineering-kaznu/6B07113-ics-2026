# Импорт библиотек
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target

print("Размер данных:", X.shape)

# 2. Проверка пропущенных значений
print("Количество пропусков:", np.isnan(X).sum())

# 3. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Нормализация признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Создание модели KNN
model = KNeighborsClassifier(n_neighbors=5)

# 6. Обучение модели
model.fit(X_train, y_train)

# 7. Предсказание
y_pred = model.predict(X_test)

# 8. Метрики качества
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("\nМетрики качества:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nПодробный отчет:")
print(classification_report(y_test, y_pred))

# 9. Кросс-валидация
cv_scores = cross_val_score(model, X, y, cv=5)

print("\nКросс-валидация (5-fold):")
print("Оценки:", cv_scores)
print("Средняя точность:", cv_scores.mean())
