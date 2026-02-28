import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================
# 1. ЗАГРУЗКА ДАННЫХ
# ==========================

df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

if "loan_id" in df.columns:
    df = df.drop(columns=["loan_id"])

# Кодирование категорий
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 2. BAGGING — Random Forest
# ==========================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

# ==========================
# 3. BOOSTING — XGBoost
# ==========================

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_proba)

# ==========================
# 4. СРАВНЕНИЕ МОДЕЛЕЙ
# ==========================

print("=== BAGGING (Random Forest) ===")
print("Accuracy:", rf_acc)
print("ROC-AUC:", rf_auc)

print("\n=== BOOSTING (XGBoost) ===")
print("Accuracy:", xgb_acc)
print("ROC-AUC:", xgb_auc)

# ==========================
# 5. ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ Accuracy и ROC-AUC
# ==========================

models = ["Random Forest (Bagging)", "XGBoost (Boosting)"]
accuracy = [rf_acc, xgb_acc]
roc_auc = [rf_auc, xgb_auc]

x = np.arange(len(models))

plt.figure(figsize=(7,5))
plt.bar(x - 0.2, accuracy, 0.4, color='skyblue')
plt.bar(x + 0.2, roc_auc, 0.4, color='orange')
plt.xticks(x, models, rotation=20)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Bagging vs Boosting — Accuracy & ROC-AUC")
plt.legend(["Accuracy", "ROC-AUC"])

# Подписи значений над столбцами
for i in range(len(models)):
    plt.text(x[i]-0.2, accuracy[i]+0.01, f"{accuracy[i]:.3f}", ha="center")
    plt.text(x[i]+0.2, roc_auc[i]+0.01, f"{roc_auc[i]:.3f}", ha="center")

plt.tight_layout()
plt.show()

# ==========================
# 6. ВИЗУАЛИЗАЦИЯ ВАЖНОСТИ ПРИЗНАКОВ
# ==========================

importances_rf = rf_model.feature_importances_
importances_xgb = xgb_model.feature_importances_
features = X.columns

# Сортировка признаков по важности Random Forest (для наглядности)
indices = np.argsort(importances_rf)[::-1]
features_sorted = features[indices]
importances_rf_sorted = importances_rf[indices]
importances_xgb_sorted = importances_xgb[indices]

x = np.arange(len(features_sorted))

plt.figure(figsize=(10,6))
plt.bar(x - 0.2, importances_rf_sorted, 0.4, color='skyblue')
plt.bar(x + 0.2, importances_xgb_sorted, 0.4, color='orange')
plt.xticks(x, features_sorted, rotation=45, ha='right')
plt.ylabel("Feature Importance")
plt.title("Feature Importance: Random Forest vs XGBoost")
plt.legend(["Random Forest (Bagging)", "XGBoost (Boosting)"])
plt.tight_layout()
plt.show()