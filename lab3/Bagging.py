# =========================
# 1. Импорт библиотек
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# 2. Загрузка данных
# =========================
df = pd.read_csv("train.csv")

# =========================
# 3. Предобработка данных
# =========================

df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Заполняем пропуски
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Кодируем категориальные признаки
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# =========================
# 4. Разделение данных
# =========================
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5. Базовая модель — Logistic Regression
# =========================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
log_acc = accuracy_score(y_test, y_pred_log)

# =========================
# 6. Bagging — Random Forest
# =========================
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

# =========================
# 7. Output (сравнение)
# =========================
print("=== Model Comparison ===")
print(f"Logistic Regression Accuracy: {log_acc:.4f}")
print(f"Random Forest Accuracy:       {rf_acc:.4f}")

# =========================
# 8. Важность признаков (Random Forest)
# =========================
plt.figure(figsize=(10, 6))
sns.barplot(x=rf_model.feature_importances_, y=X.columns)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()