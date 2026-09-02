"""
Ёмкость стратегии: сколько капитала выдерживает сигнал, прежде чем
влияние собственных заявок на цену съест преимущество.

Зачем это главный оставшийся вопрос (раздел 30.4). Весь бэктест до сих
пор считал, что наша сделка рынок не двигает: мы платим спред и комиссию,
и всё. Для маленьких денег это правда, для больших — нет: чем больше
заявка относительно оборота, тем хуже средняя цена исполнения.

Модель влияния — «закон квадратного корня», отраслевой стандарт:

    impact ≈ Y · σ · sqrt(Q / V)

где σ — волатильность бара, Q — размер заявки в рублях, V — рублёвый
оборот бара, Y ≈ 1 (берём консервативно). Платится дважды: на входе и на
выходе. Модель приближённая — но она отвечает на нужный вопрос не «сколько
именно», а «на каком порядке капитала преимущество исчезает».

ВАЖНО — стоимость пункта. В барах лежит `dollar_volume` = цена × число
контрактов, и это НЕ рубли: у CNYRUBF один контракт — 1000 юаней, у BR —
10 баррелей. Рублёвый оборот = dollar_volume × φ, где φ — стоимость
одного пункта цены:

    φ = min_price_increment_amount / min_price_increment   (из T-Invest API)

Проверено по API (2026-09-02): CNYRUBF 1 руб / 0.001 = 1000,
IMOEXF 25 / 25 = 1, SBERF 1 / 1 = 1, BR 8.56 / 0.01 = 856. Это та самая φ
из формулы ETF Trick (processing/etf_trick.py), которая для одного
инструмента сокращалась — а здесь она обязательна.

Запуск:
    .venv/bin/python -m scripts.capacity_analysis --name CNYRUBF_CONT --root CNYRUBF
    .venv/bin/python -m scripts.capacity_analysis --name SBERF_CONT --root SBERF --test
"""
import argparse

import numpy as np
import pandas as pd

from scripts.backtest import COMMISSION_BP, roll_spreads
from scripts.backtest_continuous import build_trades, load_held, load_series, net_bp
from scripts.final_eval import MKT, fit

D = "data/features/"

# φ — стоимость одного пункта цены в рублях (см. докстринг)
ROOT_TO_POINT_VALUE = {"CNYRUBF": 1000.0, "IMOEXF": 1.0, "SBERF": 1.0, "BR": 856.0}

CAPITALS = [1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9]


def attach_bar_liquidity(trades, bars, phi):
    """К каждой сделке — рублёвый оборот и волатильность бара входа/выхода."""
    # exit_time может быть naive (пришёл из ряда баров), время баров — tz-aware
    trades = trades.copy()
    for col in ("entry_time", "exit_time"):
        ts = pd.to_datetime(trades[col])
        trades[col] = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    b = bars[["time", "dollar_volume", "volatility_20"]].copy()
    b["time"] = pd.to_datetime(b["time"]).dt.tz_convert("UTC")
    b["v_rub"] = b["dollar_volume"] * phi
    for side, key in (("in", "entry_time"), ("out", "exit_time")):
        m = b.rename(columns={"time": key, "v_rub": f"v_{side}", "volatility_20": f"vol_{side}"})
        trades = trades.merge(m[[key, f"v_{side}", f"vol_{side}"]], on=key, how="left")
    # если бар выхода не найден в этой выборке — берём бар входа
    trades["v_out"] = trades["v_out"].fillna(trades["v_in"])
    trades["vol_out"] = trades["vol_out"].fillna(trades["vol_in"])
    return trades.dropna(subset=["v_in", "vol_in"])


def impact_bp(trades, q, y=1.0):
    """Влияние на цену, б.п. на сделку (вход + выход)."""
    a = y * trades["vol_in"] * np.sqrt(q / trades["v_in"]) * 1e4
    b = y * trades["vol_out"] * np.sqrt(q / trades["v_out"]) * 1e4
    return a + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="CNYRUBF_CONT")
    ap.add_argument("--root", default="CNYRUBF")
    ap.add_argument("--tag", default="_w4")
    ap.add_argument("--top-pct", type=float, default=10.0)
    ap.add_argument("--exit", dest="exit_rule", default="next-worst",
                    choices=["barrier", "next-close", "next-worst"])
    ap.add_argument("--impact-y", type=float, default=1.0, help="коэффициент закона корня")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    phi = ROOT_TO_POINT_VALUE[a.root]
    held, rolls = load_held(a.name)
    series = load_series(a.name, a.tag) if a.exit_rule != "barrier" else None
    tr = pd.read_csv(f"{D}{a.name}_train{a.tag}_v2.csv", parse_dates=["time", "t1"]).dropna(subset=MKT + ["target"])

    if a.test:
        te = pd.read_csv(f"{D}{a.name}_test{a.tag}_v2.csv", parse_dates=["time", "t1"]).dropna(subset=MKT + ["target"]).sort_values("time")
        fit_df, label = tr, "TEST"
    else:
        # последнее окно walk-forward: обучение на всём, что раньше
        t_min, t_max = tr["time"].min(), tr["time"].max()
        span = t_max - t_min
        lo = t_min + span * 5 / 6
        te = tr[tr["time"] >= lo]
        fit_df = tr[(tr["time"] < lo) & (tr["t1"] < lo) & (tr["time"] < lo - span * 0.01)]
        label = f"walk-forward, последнее окно {lo.date()}..{t_max.date()}"

    spread = roll_spreads(a.root, str(te["time"].min().date()), str(te["time"].max().date())).dropna().to_dict()
    m = fit(MKT, fit_df)
    thr = np.percentile(m.predict_proba(fit_df[MKT])[:, 1], 100 - a.top_pct)
    t = build_trades(te, m.predict_proba(te[MKT])[:, 1], thr, held, rolls, spread, 1, series, a.exit_rule)
    t = attach_bar_liquidity(t, te, phi)
    base = net_bp(t)

    day_rub = te.assign(d=te["time"].dt.date).groupby("d")["dollar_volume"].sum().mean() * phi
    print(f"=== {a.name}: ёмкость, {label} ===")
    print(f"стоимость пункта φ = {phi:g} руб; правило выхода {a.exit_rule}")
    print(f"средний дневной оборот ближнего контракта: {day_rub/1e9:.2f} млрд руб")
    print(f"медианный оборот бара: {t['v_in'].median()/1e6:.1f} млн руб; "
          f"сделок {len(t)}; чистая без влияния {base.mean():+.2f} б.п.\n")

    print(f"{'капитал в сделке':>18s} {'доля бара':>10s} {'влияние б.п.':>13s} {'чистая б.п.':>12s}")
    for q in CAPITALS:
        imp = impact_bp(t, q, a.impact_y)
        net = base - imp
        share = (q / t["v_in"]).median()
        print(f"{q/1e6:15.0f} млн {share:9.1%} {imp.mean():13.2f} {net.mean():+12.2f}")

    # где преимущество обнуляется — решаем уравнение по сетке (бинарный поиск)
    lo_q, hi_q = 1e5, 1e12
    for _ in range(60):
        mid = (lo_q * hi_q) ** 0.5
        if (base - impact_bp(t, mid, a.impact_y)).mean() > 0:
            lo_q = mid
        else:
            hi_q = mid
    print(f"\nпреимущество обнуляется на капитале ≈ {lo_q/1e6:,.0f} млн руб на сделку "
          f"({(lo_q/t['v_in'].median()):.0%} оборота бара, "
          f"{lo_q/day_rub:.1%} дневного оборота контракта)")


if __name__ == "__main__":
    main()
