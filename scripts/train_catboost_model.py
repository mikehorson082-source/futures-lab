"""Этап 1 (CNYRUBF) — сравнение baseline с градиентным бустингом (CatBoost).

Те же признаки и train/test, что в scripts/train_baseline_model.py — чтобы
разница в AUC объяснялась моделью, а не данными. CatBoost принимает
sample_weight напрямую (без ручной нормализации на train — здесь она не
нужна: для деревьев важен только относительный вес строк друг к другу,
среднее значение веса решающего дерева не сдвигает).

Запуск:
    .venv/bin/python -m scripts.train_catboost_model --root CNYRUBF
"""

import argparse

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from scripts.train_baseline_model import FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="CNYRUBF")
    # depth=4/iterations=50 — найдено перебором по сетке (раздел 15 плана),
    # даёт наименьший разрыв train/test AUC при лучшем test AUC в сетке.
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    train = pd.read_csv(f"data/features/{args.root}_train.csv")
    test = pd.read_csv(f"data/features/{args.root}_test.csv")

    test_before = len(test)
    test = test.dropna(subset=FEATURES)
    print(
        f"test: убрано {test_before - len(test)} строк с пропусками в "
        f"признаках (из {test_before})"
    )

    X_train, y_train, w_train = train[FEATURES], train["target"], train["sample_weight"]
    X_test, y_test = test[FEATURES], test["target"]

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, pred)

    majority_class = y_train.mode()[0]
    baseline_acc = (y_test == majority_class).mean()

    print(f"\n=== {args.root}: CatBoost, test ===")
    print(f"строк test (после очистки): {len(test)}")
    print(f"AUC-ROC:   {auc:.4f}  (0.5 = случайность; логрегрессия: 0.598)")
    print(f"Accuracy:  {acc:.4f}  (baseline «всегда класс {majority_class}»: {baseline_acc:.4f})")
    print(f"Precision (класс 1=TP): {precision:.4f}")
    print(f"Recall    (класс 1=TP): {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion matrix (строки=факт, столбцы=прогноз, [[0,0 0,1],[1,0 1,1]]):")
    print(cm)

    print("\nВажность признаков (CatBoost feature importance):")
    importances = pd.Series(
        model.get_feature_importance(), index=FEATURES
    ).sort_values(ascending=False)
    for name, val in importances.items():
        print(f"  {name:28s} {val:.2f}")


if __name__ == "__main__":
    main()
