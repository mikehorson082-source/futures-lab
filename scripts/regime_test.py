"""
Проверка переноса через СМЕНУ РЕЖИМА РЫНКА (пункт 29.5.1).

Всё, что сделано в проекте, измерено на 2022-2026: санкции, закрытый счёт
капитала, ключевая ставка 7.5-21%. Это один режим. Открытый вопрос: мы
нашли механизм или описали четыре конкретных года?

Данные: в БД догружены свечи 2020-2021 (T-Invest отдаёт контракты с
экспирацией не раньше 2021, поэтому реально доступен 2021 год целиком плюс
хвост 2020). Ставка ЦБ тогда была 4.25-8.5%, рынок дособытийный.

Три прогона на ОДНОМ непрерывном ряду <name> (по умолчанию SBERF_LONG):

    новый→новый  — контроль: обучение на train нового режима, проверка на
                   его же test. Должно воспроизвести известные числа.
    новый→старый — модель нового режима применена к 2021 году. Строгий
                   тест «держится ли зависимость в другом режиме».
    старый→новый — модель, обученная ТОЛЬКО на 2021, применена к test
                   нового режима. Это направление реалистично по времени
                   (учимся на прошлом, торгуем в будущем), но старого
                   периода мало — результат шумнее.

Запуск:
    .venv/bin/python -m scripts.regime_test --name SBERF_LONG --root SBERF
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from processing.features import add_derived_features
from scripts.backtest import roll_spreads
from scripts.backtest_continuous import build_trades, load_held, load_series, net_bp
from scripts.final_eval import MKT, fit

D = "data/features/"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="SBERF_LONG")
    p.add_argument("--root", default="SBERF")
    p.add_argument("--tag", default="_w4")
    p.add_argument("--regime-split", default="2022-01-01",
                   help="граница режимов: всё раньше — старый режим")
    p.add_argument("--new-test-start", default="2025-09-22",
                   help="начало test нового режима (та же граница, что в разделах 27-31)")
    p.add_argument("--top-pct", type=float, default=10.0)
    p.add_argument("--exit", dest="exit_rule", default="next-worst",
                   choices=["barrier", "next-close", "next-worst"])
    a = p.parse_args()

    full = pd.read_csv(f"{D}{a.name}_bars_labeled{a.tag}.csv", parse_dates=["time", "t1"])
    full = add_derived_features(full).dropna(subset=MKT + ["target"])
    full["time"] = pd.to_datetime(full["time"], utc=True)
    full["t1"] = pd.to_datetime(full["t1"], utc=True)

    reg = pd.Timestamp(a.regime_split, tz="UTC")
    new_test = pd.Timestamp(a.new_test_start, tz="UTC")
    old = full[full["time"] < reg]
    new_tr = full[(full["time"] >= reg) & (full["time"] < new_test) & (full["t1"] < new_test)]
    new_te = full[full["time"] >= new_test]

    print(f"=== {a.name}: тест на смену режима ===")
    for label, df in (("старый режим (до " + a.regime_split + ")", old),
                      ("новый режим, train", new_tr),
                      ("новый режим, test", new_te)):
        if not len(df):
            continue
        print(f"{label:34s} {len(df):6d} баров  {df.time.min().date()}..{df.time.max().date()}  "
              f"target=1 {df.target.mean():.1%}  медиана базиса {df.basis_pct_full.median():+.3f}%  "
              f"медиана волы {df.volatility_20.median():.4f}")

    held, rolls = load_held(a.name)
    series = load_series(a.name, a.tag) if a.exit_rule != "barrier" else None

    def run(tag, tr, te):
        if len(tr) < 1000 or len(te) < 200:
            print(f"\n--- {tag}: мало данных ({len(tr)} / {len(te)}) ---")
            return
        m = fit(MKT, tr)
        proba = m.predict_proba(te[MKT])[:, 1]
        auc = roc_auc_score(te["target"], proba)
        thr = np.percentile(m.predict_proba(tr[MKT])[:, 1], 100 - a.top_pct)
        sp = roll_spreads(a.root, str(te.time.min().date()), str(te.time.max().date())).dropna().to_dict()
        t = build_trades(te.sort_values("time"), proba, thr, held, rolls, sp, 1, series, a.exit_rule)
        n = net_bp(t) if len(t) else pd.Series(dtype=float)
        coefs = pd.Series(m.named_steps["logisticregression"].coef_[0], index=MKT)
        print(f"\n--- {tag} ---")
        print(f"  обучение {len(tr)} баров ({tr.time.min().date()}..{tr.time.max().date()}), "
              f"проверка {len(te)} баров ({te.time.min().date()}..{te.time.max().date()})")
        print(f"  AUC {auc:.3f}")
        if len(n):
            print(f"  сделок {len(t)}, чистая {n.mean():+.2f} б.п. "
                  f"(t={n.mean()/n.std()*np.sqrt(len(n)):.2f})")
        print(f"  коэффициенты: " + ", ".join(f"{k} {v:+.2f}" for k, v in coefs.items()))

    run("новый→новый (контроль)", new_tr, new_te)
    run("новый→старый (2021)", new_tr, old)
    run("старый→новый (обучение только на 2021)", old, new_te)


if __name__ == "__main__":
    main()
