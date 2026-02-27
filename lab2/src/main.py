import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 1) Загружаем CSV (после unzip он обычно называется data.csv и лежит в папке lab2/)
df = pd.read_csv("data.csv")

# Удаляем лишние столбцы, если есть
if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])
if "id" in df.columns:
    df = df.drop(columns=["id"])

# 2) Формируем X и y
y = df["diagnosis"].map({"M": 1, "B": 0})
X = df.drop(columns=["diagnosis"])

# 3) Делим на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4) Bagging: Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 5) Метрики
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("=== Bagging: RandomForest (Breast Cancer Wisconsin, Kaggle CSV) ===")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC:  {auc:.4f}")

# 6) Важность признаков (top-15)
importances = rf.feature_importances_
top = importances.argsort()[::-1][:15]

plt.figure()
plt.bar(X.columns[top], importances[top])
plt.xticks(rotation=90)
plt.title("RandomForest: Feature Importance (Top-15)")
plt.tight_layout()
plt.show()