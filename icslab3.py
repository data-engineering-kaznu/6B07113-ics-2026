import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
input_file = 'Input3.csv'
data = pd.read_csv(input_file)
data = data.drop(columns=['Id'])  
X = data.drop(columns=['quality'])
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Точность базовой модели (Decision Tree): {accuracy_dt:.2f}")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Точность ансамблевой модели (Random Forest): {accuracy_rf:.2f}")
output_file = os.path.join(os.path.dirname(input_file), 'Output3.csv')
output = X_test.copy()
output['true_quality'] = y_test
output['predicted_quality'] = y_pred_rf
output.to_csv(output_file, index=False)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(10,6))
feature_importances.plot(kind='bar')
plt.title("Важность признаков (Random Forest)")
plt.ylabel("Важность")
plt.tight_layout()
plot_file = os.path.join(os.path.dirname(input_file), 'feature_importance.png')
plt.savefig(plot_file)
plt.show()
print(f"График важности признаков сохранён в {plot_file}")
