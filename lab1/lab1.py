import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- 1. ЗАГРУЗКА И СОХРАНЕНИЕ INPUT.CSV ---
iris = load_iris()
X = iris.data
y = iris.target

# Создаем красивую таблицу для Input.csv
df_input = pd.DataFrame(X, columns=iris.feature_names)
# Добавляем колонку с названием сорта (словами, а не цифрами)
df_input['species'] = [iris.target_names[i] for i in y]

# Сохраняем входные данные в файл
df_input.to_csv('Input.csv', index=False)
print("✅ Файл Input.csv успешно создан!")

# --- 2. ПРЕДОБРАБОТКА ---
# Делим на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация (приводим к одному масштабу)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. ОБУЧЕНИЕ МОДЕЛИ (KNN) ---
k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# --- 4. ПРЕДСКАЗАНИЕ И ОЦЕНКА ---
y_pred = model.predict(X_test_scaled)

# Вывод метрик в консоль
print(f"\nТочность модели (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Построение матрицы ошибок (график)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Матрица ошибок (Confusion Matrix)")
plt.show()

# --- 5. ГЕНЕРАЦИЯ И СОХРАНЕНИЕ OUTPUT.CSV ---
# Создаем таблицу результатов
results = pd.DataFrame()

# Добавляем правильные ответы (словами)
results['True_Class'] = [iris.target_names[i] for i in y_test]

# Добавляем предсказания модели (словами)
results['Predicted_Class'] = [iris.target_names[i] for i in y_pred]

# Добавляем колонку проверки (True/False)
results['Is_Correct'] = results['True_Class'] == results['Predicted_Class']

# Сохраняем результат в файл
results.to_csv('Output.csv', index_label='id')
print("✅ Файл Output.csv успешно создан!")