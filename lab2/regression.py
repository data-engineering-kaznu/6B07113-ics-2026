import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Загрузка данных
df = pd.read_csv("housing.csv")

# 2. Удаляем категориальный столбец
df = df.drop("ocean_proximity", axis=1)

# 3. Удаляем строки с пропущенными значениями
df = df.dropna()

# 4. Признаки и цель
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# 5. Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6. Нормализация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Линейная регрессия
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Предсказание
y_pred = model.predict(X_test)

# 9. Метрики
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
