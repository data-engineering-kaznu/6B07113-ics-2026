import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
df_input = X.copy()
df_input['MedHouseVal'] = y
df_input.to_csv('Input_reg.csv', index=False)
print("✅ Input_reg.csv создан.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MSE: {mse:.4f}, R²: {r2:.4f}")
evaluate(y_test, y_pred_rf, "Random Forest")
evaluate(y_test, y_pred_xgb, "XGBoost")
df_output = pd.DataFrame({
    'Actual_Price': y_test,
    'Pred_RF': y_pred_rf,
    'Pred_XGB': y_pred_xgb,
    'Diff_RF': np.abs(y_test - y_pred_rf),
    'Diff_XGB': np.abs(y_test - y_pred_xgb)
})
df_output.to_csv('Output_reg.csv', index=False)
print("✅ Output_reg.csv создан.")
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, alpha=0.3, label='Random Forest')
plt.scatter(y_test, y_pred_xgb, alpha=0.3, label='XGBoost')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel('Реальная цена')
plt.ylabel('Предсказанная цена')
plt.title('Ансамблевые методы регрессии: Реальность vs Предсказание')
plt.legend()
plt.show()
features = X.columns
rf_importance = rf.feature_importances_
xgb_importance = xgb.feature_importances_
df_importance = pd.DataFrame({
    'Feature': features,
    'RandomForest': rf_importance,
    'XGBoost': xgb_importance
})
print("\nВажность признаков:")
print(df_importance)
df_importance.set_index('Feature').plot(kind='bar', figsize=(10,6))
plt.title('Важность признаков для Random Forest и XGBoost')
plt.ylabel('Importance')
plt.show()
