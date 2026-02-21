import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- ШАГ 1: Загрузка и сохранение Input.csv ---
housing = fetch_california_housing()
X = housing.data
y = housing.target # Цена дома в сотнях тысяч долларов

df_input = pd.DataFrame(X, columns=housing.feature_names)
df_input['MedHouseVal'] = y
df_input.to_csv('Input_reg.csv', index=False)
print("✅ Input_reg.csv создан.")

# --- ШАГ 2: Предобработка ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование (StandardScaler) здесь критически важно для Линейной регрессии
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- ШАГ 3: Обучение модели ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- ШАГ 4: Оценка качества ---
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred) # Среднеквадратичная ошибка
r2 = r2_score(y_test, y_pred)           # Коэффициент детерминации

print(f"\nMSE (Mean Squared Error): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# --- ШАГ 5: Сохранение Output.csv ---
df_output = pd.DataFrame({
    'Actual_Price': y_test,
    'Predicted_Price': y_pred,
    'Difference': np.abs(y_test - y_pred)
})
df_output.to_csv('Output_reg.csv', index=False)
print("✅ Output_reg.csv создан.")

# --- ШАГ 6: Визуализация ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Реальная цена')
plt.ylabel('Предсказанная цена')
plt.title('Линейная регрессия: Реальность vs Предсказание')
plt.show()