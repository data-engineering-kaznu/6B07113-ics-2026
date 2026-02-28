import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# --- ШАГ 1: Подготовка данных ---
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- ШАГ 2: Random Forest (Bagging) ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# --- ШАГ 3: XGBoost (Boosting) ---
# Настраиваем модель для классификации
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

# --- ШАГ 4: Сравнение результатов ---
print(f"Точность Random Forest: {accuracy_score(y_test, rf_pred):.4f}")
print(f"Точность XGBoost: {accuracy_score(y_test, xgb_pred):.4f}")

# --- ШАГ 5: Важность признаков (Feature Importance) ---
# Это ключевая фишка ансамблей — они показывают, что важнее для предсказания
importances = rf_model.feature_importances_
feature_names = iris.feature_names

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, palette="viridis", hue=feature_names, legend=False)
plt.title("Важность признаков (Random Forest)")
plt.xlabel("Уровень важности")
plt.ylabel("Признаки")
plt.show()

# Сохраняем результаты в Output_lab2.csv
output_df = pd.DataFrame({
    'Real': [iris.target_names[i] for i in y_test],
    'RF_Pred': [iris.target_names[i] for i in rf_pred],
    'XGB_Pred': [iris.target_names[i] for i in xgb_pred]
})
output_df.to_csv('Output_lab2.csv', index=False)