"""Этап 1 — проверка торговых правил walk-forward ВНУТРИ train.

Зачем отдельный скрипт: правило "торговать только ликвидные контракты"
(scripts/backtest.py) было выбрано ПОСЛЕ того, как стала видна разбивка
издержек по контрактам на test — то есть на тех же данных, на которых
потом измерялся результат. Это та же ошибка, что в разделе 16 плана.
Здесь те же правила проверяются на окнах внутри train, где test не
открывается вообще: если чистая доходность положительна и там, правило
не подогнано под test.

Дополнительно — чувствительность к проскальзыванию на стоп-лоссе: выход
"ровно по барьеру" (processing/labeling.py) для SL оптимистичен, реальный
стоп исполняется не лучше уровня.

Запуск:
    .venv/bin/python -m scripts.backtest_walkforward --root CNYRUBF
"""
import argparse

import numpy as np
import pandas as pd

from scripts.backtest import COMMISSION_BP, build_trades, roll_spreads
from scripts.final_eval import MKT, fit

TICK_BP = 0.85  # шаг цены 0.001 ₽ при цене ~11.7 ₽


def net_bp(trades, cost_map, sl_slip_ticks=0.0):
    cost = trades["ticker"].map(cost_map).fillna(np.nan)
    slip = (trades["reason"] == "sl") * sl_slip_ticks * TICK_BP
    return trades["ret_gross"] * 1e4 - cost - slip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="CNYRUBF")
    ap.add_argument("--tag", default="", help="вариант ширины барьеров, см. build_labels --tag")
    ap.add_argument("--spread-root", default=None,
                    help="по какой root-серии брать спреды и ставку из БД. Нужно для "
                         "производных выборок (например CNYRUBF_R1 — только ближний "
                         "контракт): в БД такой серии нет, спреды надо брать по CNYRUBF. "
                         "Без этого cost_map окажется пустым и издержки молча станут нулём.")
    ap.add_argument("--side", type=int, default=1, choices=[1, -1])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top-pct", type=float, default=10.0)
    ap.add_argument("--min-volume-share", type=float, default=0.5)
    ap.add_argument("--embargo-fraction", type=float, default=0.01)
    a = ap.parse_args()

    df = pd.read_csv(f"data/features/{a.root}_train{a.tag}_v2.csv", parse_dates=["time", "t1"])
    df = df.dropna(subset=MKT + ["target"])

    t_min, t_max = df["time"].min(), df["time"].max()
    span = t_max - t_min
    embargo = span * a.embargo_fraction
    edges = [t_min + span * i / (a.folds + 1) for i in range(a.folds + 2)]

    rows = []
    for k in range(1, a.folds + 1):
        lo, hi = edges[k], edges[k + 1]
        te = df[(df["time"] >= lo) & (df["time"] < hi)]
        tr = df[(df["time"] < lo) & (df["t1"] < lo) & (df["time"] < lo - embargo)]
        if len(tr) < 2000 or len(te) < 500 or te["target"].nunique() < 2:
            continue
        cost_map = (roll_spreads(a.spread_root or a.root, str(te["time"].min().date()),
                                 str(te["time"].max().date())) + COMMISSION_BP).to_dict()
        m = fit(MKT, tr)
        thr = np.percentile(m.predict_proba(tr[MKT])[:, 1], 100 - a.top_pct)
        proba = m.predict_proba(te[MKT])[:, 1]
        for tag, mvs in (("все контракты", 0.0), ("ликвидные", a.min_volume_share)):
            t = build_trades(te, proba, thr, mvs, a.side)
            if not len(t):
                continue
            n = net_bp(t, cost_map).dropna()
            n1 = net_bp(t, cost_map, sl_slip_ticks=1.0).dropna()
            rows.append({"окно": k, "период": f"{lo.date()}..{hi.date()}", "правило": tag,
                         "сделок": len(n),
                         "валовая": round(t["ret_gross"].mean() * 1e4, 2),
                         "чистая": round(n.mean(), 2),
                         "t": round(n.mean() / n.std() * np.sqrt(len(n)), 2),
                         "чистая +1 тик на SL": round(n1.mean(), 2)})

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print("\nсводка по правилам (среднее по окнам):")
    print(res.groupby("правило").agg(
        окон=("чистая", "size"), сделок=("сделок", "sum"),
        средняя_чистая=("чистая", "mean"), окон_в_плюсе=("чистая", lambda s: int((s > 0).sum())),
        с_проскальзыванием=("чистая +1 тик на SL", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
