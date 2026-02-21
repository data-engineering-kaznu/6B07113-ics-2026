import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Загрузка
df = pd.read_csv("USA Housing Dataset.csv")

# Оставляем только числовые колонки
df = df.select_dtypes(include=[np.number])

# Разделение
X = df.drop("price", axis=1)
y = df["price"]

# Нормализация
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Модель
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Метрики
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))