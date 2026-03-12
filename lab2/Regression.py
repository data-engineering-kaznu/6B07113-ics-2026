import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Загрузка данных
data = pd.read_csv("kc_house_data.csv")

# 2. Очистка и подготовка данных
# Удалим колонки, которые не нужны для модели
# 'id' уникальный идентификатор, 'date' — дата продажи
data = data.drop(columns=['id', 'date'])

# Проверка пропусков
print("Пропуски в данных:\n", data.isnull().sum())

# 3. Разделение на признаки и целевую переменную
X = data.drop("price", axis=1)  # price — наша целевая переменная
y = data["price"]

# 4. Делим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Нормализация признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Создание модели
model = LinearRegression()

# 7. Обучение модели
model.fit(X_train, y_train)

# 8. Предсказание
y_pred = model.predict(X_test)

# 9. Метрики
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMSE на тестовой выборке:", mse)
print("R2 на тестовой выборке:", r2)

# 10. Кросс-валидация (5 фолдов)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nСреднее R2 (CV):", cv_scores.mean())

# 11. Сохранение предсказаний в output.csv
output = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
output.to_csv("output.csv", index=False)

print("\nПредсказания сохранены в 'output.csv'")