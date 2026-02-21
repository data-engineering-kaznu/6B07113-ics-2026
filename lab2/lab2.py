import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #/
from sklearn.preprocessing import StandardScaler #nrm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("california_housing_train.csv")
y = df["MedHouseVal"]
X = df.drop(columns=["MedHouseVal"])
X = X.fillna(X.mean()) #nan

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42 #80%
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred) #metr
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2:", r2)

plt.figure()
plt.scatter(y_test, y_pred)
min_v = min(y_test.min(), y_pred.min())
max_v = max(y_test.max(), y_pred.max())
plt.plot([min_v, max_v], [min_v, max_v], color = "purple")
plt.xlabel("Реальные значения")
plt.ylabel("Предсказанные значения")
plt.title("Реальные vs Предсказанные")
plt.grid()
plt.show()