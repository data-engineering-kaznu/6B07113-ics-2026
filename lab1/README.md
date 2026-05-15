# Лабораторная работа №1 — Обучение с учителем: классификация и регрессия

## Цель
Изучить методы обучения с учителем на примерах:
- классификации (Iris)
- регрессии (Housing)

## Выполнено
### 1) Классификация (Iris)
- Датасет: Iris
- Предобработка: обработка пропусков (SimpleImputer), нормализация (StandardScaler)
- Модель: Logistic Regression
- Оценка: accuracy, precision, recall, F1 + 5-fold cross-validation

### 2) Регрессия (Housing)
- Датасет: California Housing
- Предобработка: обработка пропусков (SimpleImputer), нормализация (StandardScaler)
- Модель: Linear Regression
- Оценка: MSE, R² + 5-fold cross-validation

## Как запустить
1) Установить зависимости:
```bash
pip install -r requirements.txt