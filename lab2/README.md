Лабораторная работа 2 — Регрессия с California Housing
Цель

Освоить построение модели регрессии на реальном датасете, научиться предобрабатывать данные, нормализовать признаки и оценивать качество модели.

Датасет

California Housing (CSV) — данные о ценах на жильё в Калифорнии.

Признаки: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

Целевая переменная: MedHouseVal

Категориальный признак: ocean_proximity (преобразован через pd.get_dummies())

Структура проекта
lab2/
  housing.csv       # Датасет
  scr2/
    regression.py   # Скрипт с моделью регрессии

Мои действия

Скачал housing.csv и поместил его в lab2.

Создал подпапку scr2 внутри lab2 и переместил туда regression.py.

Удалил лишний файл Regression.py из lab2, чтобы остался только в scr2.

Добавил изменения в Git и запушил на GitHub:

git add housing.csv
git add scr2/regression.py
git rm lab2/Regression.py   # если остался лишний
git commit -m "Структура lab2: housing.csv в lab2, regression.py в scr2"
git push


В regression.py:

Преобразовал категориальные признаки через pd.get_dummies()

Разделил данные на обучающую и тестовую выборки

Нормализовал признаки

Построил модель линейной регрессии

Рассчитал метрики качества: MSE и R²

Датасет

Я выбрал датасет California Housing, который содержит данные о ценах на жильё в Калифорнии и различные характеристики районов.
Я скачал его в формате CSV на сайте Kaggle и сохранил на рабочем столе. Дата сэт вы можете увидеть в GitHub внутри папки lab2