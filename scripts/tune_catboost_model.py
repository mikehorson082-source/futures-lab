"""Этап 1 (CNYRUBF) — честный подбор гиперпараметров CatBoost.

Раздел 16 плана зафиксировал проблему: в предыдущем прогоне depth/iterations
подбирались по AUC на test — то есть test использовался и для выбора модели,
и для финальной проверки, а должен только для второго. Здесь это исправлено:

1. Из train (и только из train, test не открывается) вырезается validation —
   тем же способом purge/embargo, что train/test в processing/splitting.py
   (переиспользована функция напрямую, не задублирована) — потому что метки
   размечены на каждом баре и сильно перекрываются (раздел 11 плана), просто
   взять последние N% строк без purge/embargo дало бы утечку через t1.
2. Небольшой список конфигураций (см. CONFIGS) сравнивается по AUC на
   validation — test не участвует в выборе.
3. Финальная модель с лучшими параметрами переобучается на ВСЁМ train
   (train_fit + validation) и проверяется на test ОДИН раз, в конце.

Раздел 17 плана: перебор по сетке 40 точек сам создавал риск подобрать шум
на единственном validation-разрезе (ландшафт AUC по сетке почти плоский —
40 попыток "угадать" почти неотличимы друг от друга). Здесь сетка сужена
до 3 ЗАРАНЕЕ выбранных, качественно разных по сложности конфигураций
(не мелкая перенастройка соседних точек) — меньше конфигураций для
сравнения снижает шанс того, что "победитель" на validation — просто
шум выбора, а не реальное преимущество.

Запуск:
    .venv/bin/python -m scripts.tune_catboost_model --root CNYRUBF
"""

import argparse
from datetime import datetime

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from processing.splitting import compute_split_time, purge_embargo_split
from scripts.train_baseline_model import FEATURES

# 3 заранее выбранные, качественно разные по сложности конфигурации
# (не мелкая сетка соседних точек — раздел 17 плана про риск подобрать шум
# на единственном validation-разрезе при широком переборе).
CONFIGS = [
    (2, 50),    # минимальная сложность, почти линейная модель
    (4, 100),   # умеренная
    (6, 500),   # исходная, без регуляризации (раздел 15 плана)
]


def _fit_eval(X_tr, y_tr, w_tr, X_val, y_val, depth, iterations):
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="CNYRUBF")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--embargo-fraction", type=float, default=0.01)
    args = parser.parse_args()

    train_full = pd.read_csv(f"data/features/{args.root}_train.csv")
    test = pd.read_csv(f"data/features/{args.root}_test.csv").dropna(subset=FEATURES)

    # --- Шаг 1: вырезаем validation ИЗ TRAIN тем же purge/embargo, что train/test ---
    rows = train_full.to_dict("records")
    t_min = min(datetime.fromisoformat(r["time"]) for r in rows)
    t_max = max(datetime.fromisoformat(r["time"]) for r in rows)
    split_time = compute_split_time(t_min, t_max, test_fraction=args.val_fraction)
    train_fit_rows, val_rows, report = purge_embargo_split(
        rows, split_time, t_min, t_max, embargo_fraction=args.embargo_fraction
    )
    print(
        f"validation вырезан из train: fit={report['n_train']}, "
        f"val={report['n_test']}, purge={report['n_purged']}, "
        f"embargo={report['n_embargoed']} (test.csv не открывался)"
    )

    train_fit = pd.DataFrame(train_fit_rows)
    val = pd.DataFrame(val_rows)

    X_fit, y_fit, w_fit = train_fit[FEATURES], train_fit["target"], train_fit["sample_weight"]
    X_val, y_val = val[FEATURES], val["target"]

    # --- Шаг 2: 3 конфигурации по AUC на validation, test не участвует ---
    print(f"\n{'depth':>5} {'iters':>6} {'val_auc':>8}")
    results = []
    for depth, iterations in CONFIGS:
        auc = _fit_eval(X_fit, y_fit, w_fit, X_val, y_val, depth, iterations)
        results.append((depth, iterations, auc))
        print(f"{depth:>5} {iterations:>6} {auc:>8.4f}")

    best_depth, best_iterations, best_val_auc = max(results, key=lambda r: r[2])
    print(
        f"\nЛучшая точка по validation: depth={best_depth}, "
        f"iterations={best_iterations}, val AUC={best_val_auc:.4f}"
    )

    # --- Шаг 3: финальная модель на ВСЁМ train, test открывается один раз ---
    X_train, y_train, w_train = (
        train_full[FEATURES],
        train_full["target"],
        train_full["sample_weight"],
    )
    X_test, y_test = test[FEATURES], test["target"]

    final_model = CatBoostClassifier(
        iterations=best_iterations,
        depth=best_depth,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )
    final_model.fit(X_train, y_train, sample_weight=w_train)
    test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])

    print(f"\n=== Финальная проверка на test (один раз) ===")
    print(f"depth={best_depth}, iterations={best_iterations}")
    print(f"val AUC:  {best_val_auc:.4f}")
    print(f"test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
