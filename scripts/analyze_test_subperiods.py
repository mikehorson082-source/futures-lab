"""Этап 1 (CNYRUBF) — AUC по под-периодам test: проверка гипотезы о дрейфе
режима (раздел 18 плана).

Обучает логрегрессию и CatBoost (depth=6/iter=500 — конфигурация, которая
выигрывала validation, но проваливалась на test целиком, разделы 15/17-18)
на всём train, затем считает AUC ОТДЕЛЬНО по календарным бакетам test —
если AUC систематически падает по мере углубления в test, это говорит в
пользу продолжающегося дрейфа, а не одного разового разрыва между
train и test.

Запуск:
    .venv/bin/python -m scripts.analyze_test_subperiods --root CNYRUBF
"""

import argparse

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.train_baseline_model import FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="CNYRUBF")
    parser.add_argument("--n-buckets", type=int, default=5)
    args = parser.parse_args()

    train = pd.read_csv(f"data/features/{args.root}_train.csv")
    test = pd.read_csv(f"data/features/{args.root}_test.csv").dropna(subset=FEATURES)
    test["time"] = pd.to_datetime(test["time"])

    X_train, y_train, w_train = train[FEATURES], train["target"], train["sample_weight"]
    X_test = test[FEATURES]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_s, y_train, sample_weight=w_train / w_train.mean())
    test["proba_logreg"] = logreg.predict_proba(scaler.transform(X_test))[:, 1]

    cat = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=False,
    )
    cat.fit(X_train, y_train, sample_weight=w_train)
    test["proba_cat"] = cat.predict_proba(X_test)[:, 1]

    # Календарные бакеты по времени бара (не по числу строк — те же
    # соображения о неравномерной плотности, что в processing/splitting.py)
    test["bucket"] = pd.cut(test["time"], bins=args.n_buckets)

    print(f"=== {args.root}: AUC по {args.n_buckets} под-периодам test ===\n")
    header = f"{'период':^38} {'N':>7} {'log_ret_auc':>11} {'cat_auc':>8} {'mean|basis_full|':>17}"
    print(header)
    for bucket, g in test.groupby("bucket", observed=True):
        if len(g) < 30 or g["target"].nunique() < 2:
            print(f"{str(bucket):38} {len(g):>7}  (пропущен — мало данных/один класс)")
            continue
        auc_log = roc_auc_score(g["target"], g["proba_logreg"])
        auc_cat = roc_auc_score(g["target"], g["proba_cat"])
        mean_abs_basis = g["basis_pct_full"].abs().mean()
        print(f"{str(bucket):38} {len(g):>7} {auc_log:>11.4f} {auc_cat:>8.4f} {mean_abs_basis:>17.3f}")

    print("\nСправочно — AUC на всём test:")
    print(f"  логрегрессия: {roc_auc_score(test['target'], test['proba_logreg']):.4f}")
    print(f"  CatBoost:     {roc_auc_score(test['target'], test['proba_cat']):.4f}")


if __name__ == "__main__":
    main()
