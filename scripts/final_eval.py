"""Этап 1 — ФИНАЛЬНЫЙ прогон на test. Открывается один раз.

Конфигурация выбрана заранее по walk-forward внутри train
(scripts/walkforward_eval.py): 6 стационарных рыночных признаков +
логрегрессия. Здесь ничего не подбирается — только измеряется.

Что считается, кроме одного числа AUC:
  * блочный бутстрап-доверительный интервал — насколько результат вообще
    отличим от случайности с учётом того, что соседние метки перекрываются
    (обычный построчный бутстрап это перекрытие проигнорировал бы и дал бы
    слишком узкий, обманчиво уверенный интервал);
  * AUC по календарным под-периодам test — прямое сравнение с разделом 19;
  * решающий порог, выбранный НА TRAIN (топ-N% вероятностей), и precision/
    lift на test при этом пороге — раздел 14/16: порог 0.5 на
    несбалансированных классах бесполезен, но это вопрос порога, не модели.

Запуск:
    .venv/bin/python -m scripts.final_eval --root CNYRUBF
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

V1 = ["days_to_expiration", "life_fraction_remaining", "contract_rank", "volume_share",
      "basis_pct_rub_only", "basis_pct_full", "log_return_1", "volatility_20", "momentum_10"]
MKT = ["basis_z", "basis_chg_20", "vol_ratio", "ret_1_z", "mom_10_z", "mom_50_z"]


def fit(F, tr):
    trf = tr.dropna(subset=F + ["target"])
    w = trf["sample_weight"] / trf["sample_weight"].mean()
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    m.fit(trf[F], trf["target"], logisticregression__sample_weight=w)
    return m


def block_bootstrap_auc(y, p, block=500, n_boot=1000, seed=0):
    """Бутстрап блоками подряд идущих строк: соседние метки перекрываются
    (mean overlap 7.8, раздел 11), независимыми строки считать нельзя."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    starts = np.arange(0, len(y) - block)
    k = len(y) // block
    out = []
    for _ in range(n_boot):
        s = rng.choice(starts, size=k)
        idx = (s[:, None] + np.arange(block)[None, :]).ravel()
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, p[idx]))
    return np.percentile(out, [2.5, 50, 97.5])


def main():
    p_ = argparse.ArgumentParser()
    p_.add_argument("--root", default="CNYRUBF")
    p_.add_argument("--tag", default="", help="вариант ширины барьеров, см. build_labels --tag")
    p_.add_argument("--top-pct", type=float, default=10.0)
    a = p_.parse_args()

    tr = pd.read_csv(f"data/features/{a.root}_train{a.tag}_v2.csv", parse_dates=["time"])
    te = pd.read_csv(f"data/features/{a.root}_test{a.tag}_v2.csv", parse_dates=["time"]).sort_values("time")

    for tag, F in (("v1 (уровни, разделы 14-19)", V1), ("v2 (стационарные рыночные)", MKT)):
        m = fit(F, tr)
        tef = te.dropna(subset=F + ["target"]).copy()
        tef["p"] = m.predict_proba(tef[F])[:, 1]
        auc = roc_auc_score(tef["target"], tef["p"])
        lo, med, hi = block_bootstrap_auc(tef["target"], tef["p"])
        print(f"\n=== {tag} — test, {len(tef)} строк ===")
        print(f"AUC {auc:.4f}   блочный бутстрап 95% ДИ [{lo:.4f}, {hi:.4f}]")

        # под-периоды: те же 5 календарных бакетов, что в разделе 19
        edges = pd.date_range(tef["time"].min(), tef["time"].max(), periods=6)
        rows = []
        for i in range(5):
            b = tef[(tef["time"] >= edges[i]) & (tef["time"] <= edges[i + 1])]
            if b["target"].nunique() < 2:
                continue
            rows.append({
                "период": f"{edges[i].date()}..{edges[i+1].date()}", "N": len(b),
                "AUC": round(roc_auc_score(b["target"], b["p"]), 3),
            })
        print(pd.DataFrame(rows).to_string(index=False))

        if F is MKT:
            # порог берём НА TRAIN (квантиль вероятностей train), не на test
            trf = tr.dropna(subset=F + ["target"])
            thr = np.percentile(m.predict_proba(trf[F])[:, 1], 100 - a.top_pct)
            sel = tef[tef["p"] >= thr]
            base = tef["target"].mean()
            print(f"\nпорог с train (топ {a.top_pct:.0f}% вероятностей) = {thr:.4f}")
            print(f"  отобрано на test: {len(sel)} строк ({len(sel)/len(tef):.1%})")
            print(f"  precision (доля TP): {sel['target'].mean():.4f}  против базовой {base:.4f}"
                  f"  → lift {sel['target'].mean()/base:.2f}x")

        coefs = pd.Series(m.named_steps["logisticregression"].coef_[0], index=F)
        print("\nкоэффициенты:", ", ".join(f"{k} {v:+.3f}" for k, v in
              coefs.sort_values(key=abs, ascending=False).items()))


if __name__ == "__main__":
    main()
