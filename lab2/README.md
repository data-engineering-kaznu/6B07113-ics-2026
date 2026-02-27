# ЛЗ 2. Предсказательная аналитика на основе ансамблевого обучения

## Цель
Изучить концепцию ансамблевых методов (Bagging, Boosting) и применить их для повышения точности предсказаний. Реализовать Random Forest (Bagging) и XGBoost/LightGBM (Boosting), сравнить с базовыми моделями, визуализировать важность признаков.

## Теория (кратко)
Ансамбль — комбинация нескольких моделей для получения более сильной модели.

**Bagging** (bootstrap aggregating): обучаем много моделей независимо на случайных подвыборках данных и агрегируем предсказания (усреднение/голосование). Пример: Random Forest.

**Boosting**: обучаем модели последовательно; каждая следующая модель исправляет ошибки предыдущих. Примеры: AdaBoost, Gradient Boosting, XGBoost, LightGBM.

Преимущества:
- снижение переобучения (особенно у bagging),
- повышение устойчивости к шуму,
- часто лучшее качество на табличных данных.

## Данные
Использован встроенный датасет `Breast Cancer Wisconsin` из scikit-learn (бинарная классификация).

## Модели
Базовые:
- Logistic Regression
- Decision Tree

Ансамбли:
- RandomForestClassifier (Bagging)
- XGBClassifier (Boosting)
- LGBMClassifier (Boosting)

## Метрики
- Accuracy
- F1-score
- ROC-AUC

## Визуализация
Сохранены top-15 графиков важности признаков для RF, XGBoost, LightGBM в `outputs/`.

## Как запустить
```bash
cd lab2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py