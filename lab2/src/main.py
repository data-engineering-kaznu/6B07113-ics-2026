import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# 1) Загрузка данных
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

# Функция оценки модели
def evaluate(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

    return {"Model": name, "Accuracy": acc, "F1": f1, "ROC_AUC": auc}

results = []

# 4) Базовая модель
baseline = LogisticRegression(max_iter=5000)
results.append(evaluate(baseline, "Baseline: LogisticRegression"))

# 5) Bagging — RandomForest
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
results.append(evaluate(rf, "Bagging: RandomForest"))

# 6) Boosting — GradientBoosting
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
results.append(evaluate(gb, "Boosting: GradientBoosting"))

# 7) Таблица сравнения
res_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
print("\n=== Сравнение моделей (по ROC-AUC) ===")
print(res_df.to_string(index=False))

# 8) Важность признаков (Gradient Boosting)
importances = gb.feature_importances_
top = importances.argsort()[::-1][:15]

plt.figure()
plt.bar(X.columns[top], importances[top])
plt.xticks(rotation=90)
plt.title("GradientBoosting: Feature Importance (Top-15)")
plt.tight_layout()
plt.show()
