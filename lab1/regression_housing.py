import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # 1) Загрузка данных (California Housing)
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    # 2) Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Предобработка + модель (в пайплайне)
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])

    # 4) Обучение
    model.fit(X_train, y_train)

    # 5) Предсказание
    y_pred = model.predict(X_test)

    # 6) Метрики
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== РЕГРЕССИЯ: California Housing ===")
    print(f"MSE: {mse:.4f}")
    print(f"R^2: {r2:.4f}")

    # 7) Кросс-валидация
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"neg_mse": "neg_mean_squared_error", "r2": "r2"}
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring)

    mse_scores = -cv_res["test_neg_mse"]   # переводим из отрицательного
    r2_scores = cv_res["test_r2"]

    print("=== КРОСС-ВАЛИДАЦИЯ (5-fold) ===")
    print(f"MSE: mean={mse_scores.mean():.4f}, std={mse_scores.std():.4f}")
    print(f"R^2: mean={r2_scores.mean():.4f}, std={r2_scores.std():.4f}")


if __name__ == "__main__":
    main()

