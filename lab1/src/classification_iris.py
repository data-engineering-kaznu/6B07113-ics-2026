import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def main():
    # 1) Загрузка данных (Iris)
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # 2) Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Предобработка + модель (в пайплайне)
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 4) Обучение
    model.fit(X_train, y_train)

    # 5) Предсказание
    y_pred = model.predict(X_test)

    # 6) Метрики
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("=== КЛАССИФИКАЦИЯ: Iris ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (macro)")
    print(f"Recall   : {rec:.4f} (macro)")
    print(f"F1-score : {f1:.4f} (macro)")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 7) Кросс-валидация
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro"
    }
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring)

    print("=== КРОСС-ВАЛИДАЦИЯ (5-fold) ===")
    for k in scoring.keys():
        scores = cv_res[f"test_{k}"]
        print(f"{k}: mean={scores.mean():.4f}, std={scores.std():.4f}")


if __name__ == "__main__":
    main()
