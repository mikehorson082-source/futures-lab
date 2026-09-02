"""Этап 1 — walk-forward оценка наборов признаков и моделей ВНУТРИ train.

Зачем: разделы 15-18 плана показали, что один разрез (или один
validation-срез) не даёт уверенности — выигрыш в 0.02 AUC на одном куске
времени неотличим от шума выбора. Здесь train режется на несколько
последовательных окон; для каждого окна модель учится на ВСЁМ, что было
строго раньше (расширяющееся окно — так же работала бы живая система), с
purge по t1 и embargo. Итог — не одно число, а AUC по каждому окну:
среднее, разброс и в скольких окнах вариант выиграл.

Test не открывается вообще — он остаётся для одного финального прогона.

Запуск:
    .venv/bin/python -m scripts.walkforward_eval --root CNYRUBF
"""
import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

V1 = ["days_to_expiration", "life_fraction_remaining", "contract_rank", "volume_share",
      "basis_pct_rub_only", "basis_pct_full", "log_return_1", "volatility_20", "momentum_10"]
STRUCT = ["life_fraction_remaining", "contract_rank", "volume_share"]
MKT = ["basis_z", "basis_chg_20", "vol_ratio", "ret_1_z", "mom_10_z", "mom_50_z"]

# Заранее зафиксированный, узкий список вариантов (урок раздела 18: широкая
# сетка сама подбирает шум). Каждый вариант отвечает на конкретный вопрос.
ARMS = {
    "v1 + логрегрессия":            (V1,           "logreg"),
    "v2 (структ+рынок) + логрегр.": (STRUCT + MKT, "logreg"),
    "v2 только рынок + логрегр.":   (MKT,          "logreg"),
    "v2 только рынок + CatBoost":   (MKT,          "catboost"),
}


# Наборы для серий БЕЗ внешнего спота (BR, IMOEXF, SBERF): базиса там нет
# вообще, зато есть carry_annual — цена времени, снятая с самой кривой
# (scripts/build_continuous_features.py). Включаются флагом --with-carry.
NOBASIS = ["vol_ratio", "ret_1_z", "mom_10_z", "mom_50_z"]
CARRY = ["carry_z", "carry_chg_20"]
ARMS_CARRY = {
    "рынок без базиса + логрегр.":     (NOBASIS,          "logreg"),
    "рынок + carry + логрегр.":        (NOBASIS + CARRY,  "logreg"),
    "рынок + carry + CatBoost":        (NOBASIS + CARRY,  "catboost"),
    "только carry + логрегр.":         (CARRY,            "logreg"),
}


def make_model(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return CatBoostClassifier(depth=4, iterations=100, learning_rate=0.05, verbose=0)


def fit_predict(kind, Xtr, ytr, w, Xte):
    m = make_model(kind)
    if kind == "logreg":
        m.fit(Xtr, ytr, logisticregression__sample_weight=w)
    else:
        m.fit(Xtr, ytr, sample_weight=w)
    return m.predict_proba(Xte)[:, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="CNYRUBF")
    p.add_argument("--tag", default="", help="вариант ширины барьеров, см. build_labels --tag")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--with-carry", action="store_true",
                   help="сравнивать наборы с carry вместо наборов с базисом "
                        "(для серий, где спота нет: BR, IMOEXF, SBERF)")
    p.add_argument("--embargo-fraction", type=float, default=0.01)
    a = p.parse_args()

    df = pd.read_csv(f"data/features/{a.root}_train{a.tag}_v2.csv", parse_dates=["time", "t1"])
    t_min, t_max = df["time"].min(), df["time"].max()
    span = t_max - t_min
    embargo = span * a.embargo_fraction

    # train режем на folds+1 равных КАЛЕНДАРНЫХ кусков: первый — стартовый
    # обучающий минимум, остальные folds штук по очереди играют роль теста.
    edges = [t_min + span * i / (a.folds + 1) for i in range(a.folds + 2)]

    arms = ARMS_CARRY if a.with_carry else ARMS
    results = {name: [] for name in arms}
    print(f"train: {len(df)} строк, {t_min.date()} .. {t_max.date()}, окон {a.folds}\n")

    for k in range(1, a.folds + 1):
        lo, hi = edges[k], edges[k + 1]
        te = df[(df["time"] >= lo) & (df["time"] < hi)]
        # purge по фактическим окнам меток + embargo перед началом окна
        tr = df[(df["time"] < lo) & (df["t1"] < lo) & (df["time"] < lo - embargo)]
        print(f"окно {k}: {lo.date()}..{hi.date()}  fit {len(tr)} / eval {len(te)}", end="")
        if len(tr) < 2000 or len(te) < 500 or te["target"].nunique() < 2:
            print("  — пропущено (мало данных)")
            continue
        print()
        w_all = tr["sample_weight"] / tr["sample_weight"].mean()
        for name, (F, kind) in arms.items():
            trf = tr.dropna(subset=F + ["target"])
            tef = te.dropna(subset=F + ["target"])
            proba = fit_predict(kind, trf[F], trf["target"], w_all.loc[trf.index], tef[F])
            auc = roc_auc_score(tef["target"], proba)
            results[name].append(auc)
            print(f"    {name:32s} AUC {auc:.4f}")

    print("\n=== итог по всем окнам ===")
    n = min(len(v) for v in results.values())
    mat = pd.DataFrame({k: v[:n] for k, v in results.items()})
    wins = mat.idxmax(axis=1).value_counts()
    summary = pd.DataFrame({
        "mean AUC": mat.mean(), "std": mat.std(), "min": mat.min(), "max": mat.max(),
        "побед окон": [int(wins.get(c, 0)) for c in mat.columns],
    }).sort_values("mean AUC", ascending=False)
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
