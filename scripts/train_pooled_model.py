"""
Эксперимент «одна модель на все фьючерсы» — обучение и честное сравнение.

Сравниваем на ОДНИХ И ТЕХ ЖЕ test-строках четыре режима:

    own    — модель обучена только на своём инструменте (как в Этапе 1);
    pool   — одна модель на объединённом train всех четырёх серий,
             БЕЗ признака "какой это инструмент" (проверка: есть ли общий,
             переносимый механизм);
    pool+id— то же, но root_symbol подан как категориальный признак
             (модель знает, на каком рынке она стоит, но учится на всех);
    loio   — leave-one-instrument-out: обучена на ТРЁХ остальных сериях и
             применена к четвёртой, которую не видела ни одного бара.
             Самый жёсткий тест переносимости.

Веса: sample_weight (уникальность по AFML) нормируется на среднее ВНУТРИ
каждой серии. Иначе серия с более частым нахлёстом меток (у BR mean-вес
0.080, у CNYRUBF 0.093) получила бы систематически меньший вклад просто
из-за структуры разметки, а не из-за полезности.

Запуск:
    .venv/bin/python -m scripts.train_pooled_model --tag _w4
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path(__file__).resolve().parent.parent / "data" / "features"

# Признаки, общие для ВСЕХ серий: базис считается только для CNYRUBF
# (ROOT_TO_SPOT_PAIR), поэтому basis_* и basis_z из пула исключены — иначе
# модель училась бы на колонке, которая у трёх серий из четырёх пустая.
FEATURES = [
    "days_to_expiration", "life_fraction_remaining", "contract_rank", "volume_share",
    "log_return_1", "volatility_20", "momentum_10",
    "vol_ratio", "ret_1_z", "mom_10_z", "mom_50_z",
]


def fit(train, features, cat_features=None, depth=4, iterations=50, seed=42):
    m = CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC", random_seed=seed,
        verbose=False, cat_features=cat_features or [],
    )
    m.fit(train[features], train["target"], sample_weight=train["w"])
    return m


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="_w4")
    p.add_argument("--pool", default="POOL")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--iterations", type=int, default=50)
    a = p.parse_args(argv)

    train = pd.read_csv(DATA / f"{a.pool}_train{a.tag}.csv")
    test = pd.read_csv(DATA / f"{a.pool}_test{a.tag}.csv")

    for df in (train, test):
        df.dropna(subset=FEATURES + ["target"], inplace=True)
    # вес: нормировка на среднее внутри серии (см. docstring)
    train["w"] = train["sample_weight"] / train.groupby("root_symbol")["sample_weight"].transform("mean")
    test["w"] = 1.0

    roots = sorted(train["root_symbol"].unique())
    print(f"train {len(train)} строк, test {len(test)} строк, серии: {', '.join(roots)}\n")

    results = {}

    # --- pool и pool+id
    m_pool = fit(train, FEATURES, depth=a.depth, iterations=a.iterations)
    m_pool_id = fit(train, FEATURES + ["root_symbol"], cat_features=["root_symbol"],
                    depth=a.depth, iterations=a.iterations)

    for root in roots:
        tr_r = train[train.root_symbol == root]
        te_r = test[test.root_symbol == root]
        y = te_r["target"]
        row = {"n_train": len(tr_r), "n_test": len(te_r), "base_rate": y.mean()}

        m_own = fit(tr_r, FEATURES, depth=a.depth, iterations=a.iterations)
        row["own"] = roc_auc_score(y, m_own.predict_proba(te_r[FEATURES])[:, 1])
        row["pool"] = roc_auc_score(y, m_pool.predict_proba(te_r[FEATURES])[:, 1])
        row["pool+id"] = roc_auc_score(y, m_pool_id.predict_proba(te_r[FEATURES + ["root_symbol"]])[:, 1])

        others = train[train.root_symbol != root]
        m_loio = fit(others, FEATURES, depth=a.depth, iterations=a.iterations)
        row["loio"] = roc_auc_score(y, m_loio.predict_proba(te_r[FEATURES])[:, 1])
        results[root] = row
        print(f"{root} готово")

    res = pd.DataFrame(results).T
    print("\n=== test AUC на одних и тех же строках ===")
    print(f"{'серия':9s} {'n_train':>8s} {'n_test':>7s} {'target=1':>9s} "
          f"{'own':>7s} {'pool':>7s} {'pool+id':>8s} {'loio':>7s}")
    for r, v in res.iterrows():
        print(f"{r:9s} {int(v.n_train):8d} {int(v.n_test):7d} {v.base_rate:8.1%} "
              f"{v['own']:7.3f} {v['pool']:7.3f} {v['pool+id']:8.3f} {v['loio']:7.3f}")
    print(f"\nсредний AUC по сериям (равный вес инструментов): "
          f"own {res['own'].mean():.3f}  pool {res['pool'].mean():.3f}  "
          f"pool+id {res['pool+id'].mean():.3f}  loio {res['loio'].mean():.3f}")

    print("\nВажность признаков (pool):")
    imp = pd.Series(m_pool.get_feature_importance(), index=FEATURES).sort_values(ascending=False)
    for k, v in imp.items():
        print(f"  {k:24s} {v:6.2f}")

    out = DATA / f"{a.pool}_auc_comparison{a.tag}.csv"
    res.to_csv(out)
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
