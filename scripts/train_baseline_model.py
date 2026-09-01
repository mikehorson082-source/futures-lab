"""Первая модель Этапа 1 (CNYRUBF) — baseline: логистическая регрессия.

Обучается на data/features/CNYRUBF_train.csv, проверяется на
data/features/CNYRUBF_test.csv. Признаки — только раздел 10.2 плана
(структурные, базис, ценовые), без утечки через exit_price/exit_reason/
bars_held/ret_gross/t1 (результат разметки, не признак на момент бара).

Запуск:
    .venv/bin/python -m scripts.train_baseline_model --root CNYRUBF
"""

import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "days_to_expiration",
    "life_fraction_remaining",
    "contract_rank",
    "volume_share",
    "basis_pct_rub_only",
    "basis_pct_full",
    "log_return_1",
    "volatility_20",
    "momentum_10",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="CNYRUBF")
    args = parser.parse_args()

    train = pd.read_csv(f"data/features/{args.root}_train.csv")
    test = pd.read_csv(f"data/features/{args.root}_test.csv")

    test_before = len(test)
    test = test.dropna(subset=FEATURES)
    print(
        f"test: убрано {test_before - len(test)} строк с пропусками в "
        f"признаках (из {test_before})"
    )

    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    # Нормализация веса на train (среднее = 1) — раздел 13.3 плана: вес
    # хранится ненормализованным, нормализация — на train в момент обучения.
    w_train = train["sample_weight"] / train["sample_weight"].mean()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train, sample_weight=w_train)

    proba = model.predict_proba(X_test_s)[:, 1]
    pred = model.predict(X_test_s)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, pred)

    majority_class = y_train.mode()[0]
    baseline_acc = (y_test == majority_class).mean()

    print(f"\n=== {args.root}: логистическая регрессия, test ===")
    print(f"строк test (после очистки): {len(test)}")
    print(f"AUC-ROC:   {auc:.4f}  (0.5 = случайность)")
    print(f"Accuracy:  {acc:.4f}  (baseline «всегда класс {majority_class}»: {baseline_acc:.4f})")
    print(f"Precision (класс 1=TP): {precision:.4f}")
    print(f"Recall    (класс 1=TP): {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion matrix (строки=факт, столбцы=прогноз, [[0,0 0,1],[1,0 1,1]]):")
    print(cm)

    print("\nКоэффициенты модели (после стандартизации — сравнимы по величине):")
    coefs = pd.Series(model.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
    for name, val in coefs.items():
        print(f"  {name:28s} {val:+.4f}")


if __name__ == "__main__":
    main()
