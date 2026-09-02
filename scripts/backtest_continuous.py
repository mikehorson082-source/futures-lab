"""
Бэктест на НЕПРЕРЫВНОМ ряду (ETF Trick, раздел 23 журнала) — с издержками
удерживаемого контракта И с явной издержкой каждого ролла.

Чем отличается от scripts/backtest.py (пул контрактов):

1. Цена сделки — NAV, а не котировка контракта. Значит потеря на контанго
   (≈10.6% годовых, раздел 23.2) уже внутри валовой доходности сделки, её
   не надо вспоминать отдельно.
2. Спред берётся по РЕАЛЬНО удерживаемому контракту на момент входа и
   выхода (сайдкар <name>_held.csv), а не по имени производной серии.
3. **Издержки ролла считаются явно.** Если позиция пережила переход между
   контрактами, за него заплачено: половина спреда старого + половина
   спреда нового + комиссия. В пуле этой статьи расходов не было вообще —
   там позиция жила внутри одного контракта и роллов не видела, хотя в
   реальности удержание позиции через экспирацию без ролла невозможно.

По умолчанию — walk-forward ВНУТРИ train (test не открывается), как в
scripts/backtest_walkforward.py. Test — только явным флагом --test.

Запуск:
    .venv/bin/python -m scripts.backtest_continuous --name CNYRUBF_CONT --tag _w4
    .venv/bin/python -m scripts.backtest_continuous --name CNYRUBF_CONT --tag _w4 --test
"""
import argparse

import numpy as np
import pandas as pd

from scripts.backtest import COMMISSION_BP, roll_spreads, stats
from scripts.backtest_walkforward import TICK_BP
from scripts.compute_basis import get_rate_series, rate_as_of
from scripts.final_eval import MKT, fit
from scripts.walkforward_eval import CARRY, NOBASIS

D = "data/features/"


def _utc(ts):
    """Timestamp -> naive datetime64[ns] в UTC (numpy не умеет таймзоны)."""
    t = pd.Timestamp(ts)
    return np.datetime64(t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t)


def load_held(name):
    h = pd.read_csv(f"{D}{name}_held.csv", parse_dates=["time"]).sort_values("time")
    chg = h["held_ticker"] != h["held_ticker"].shift()
    rolls = h[chg].iloc[1:]  # первый бар — не ролл
    prev = h["held_ticker"].shift()[chg].iloc[1:]
    rolls = pd.DataFrame({"time": rolls["time"].values, "from": prev.values,
                          "to": rolls["held_ticker"].values})
    return h, rolls


def build_trades(te, proba, thr, held, rolls, spread, side=1):
    """Одна позиция за раз, вход по открытию следующего бара, издержки —
    по удерживаемому контракту плюс роллы внутри сделки."""
    d = te.copy()
    d["p"] = proba
    d = d.sort_values("time")
    d["entry_price"] = d["open"].shift(-1)
    d["entry_time"] = d["time"].shift(-1)
    d = d.dropna(subset=["entry_price", "entry_time", "exit_price", "t1"])
    d = d[d["p"] >= thr].sort_values("entry_time")

    held_times = np.array([_utc(t) for t in held["time"]])
    held_tick = held["held_ticker"].values
    roll_times = np.array([_utc(t) for t in rolls["time"]]) if len(rolls) else np.array([], dtype="datetime64[ns]")
    med = float(np.nanmedian(list(spread.values()))) if spread else 0.0
    sp = lambda t: spread.get(t, med)

    def ticker_at(ts):
        i = min(np.searchsorted(held_times, _utc(ts), side="right"), len(held_tick)) - 1
        return held_tick[max(i, 0)]

    trades, free_at = [], None
    for r in d.itertuples():
        if free_at is not None and r.entry_time < free_at:
            continue
        t_in, t_out = ticker_at(r.entry_time), ticker_at(r.t1)
        lo = np.searchsorted(roll_times, _utc(r.entry_time), side="right")
        hi = np.searchsorted(roll_times, _utc(r.t1), side="right")
        roll_cost = 0.0
        for j in range(lo, hi):
            roll_cost += 0.5 * sp(rolls["from"].iloc[j]) + 0.5 * sp(rolls["to"].iloc[j]) + COMMISSION_BP
        trades.append({
            "entry_time": r.entry_time, "exit_time": r.t1, "reason": r.exit_reason,
            "ret_gross": side * (r.exit_price - r.entry_price) / r.entry_price,
            "cost_bp": 0.5 * sp(t_in) + 0.5 * sp(t_out) + COMMISSION_BP + roll_cost,
            "roll_cost_bp": roll_cost, "n_rolls": hi - lo,
        })
        free_at = r.t1
    return pd.DataFrame(trades)


def net_bp(t, sl_slip_ticks=0.0):
    slip = (t["reason"] == "sl") * sl_slip_ticks * TICK_BP
    return t["ret_gross"] * 1e4 - t["cost_bp"] - slip


def equity(trades, days, rf_daily):
    if not len(trades):
        return (1 + rf_daily).cumprod(), 0.0
    net = (net_bp(trades) / 1e4).values
    exit_day = pd.to_datetime(trades["exit_time"]).dt.tz_localize(None).dt.normalize().values
    in_pos = np.zeros(len(days), dtype=bool)
    for t in trades.itertuples():
        a = pd.Timestamp(t.entry_time).tz_localize(None).normalize()
        b = pd.Timestamp(t.exit_time).tz_localize(None).normalize()
        in_pos[(days >= a) & (days <= b)] = True
    cap, out = 1.0, []
    for i, day in enumerate(days):
        if not in_pos[i]:
            cap *= (1 + rf_daily.iloc[i])
        for r in net[exit_day == day.to_datetime64()]:
            cap *= (1 + r)
        out.append(cap)
    return pd.Series(out, index=days), in_pos.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="CNYRUBF_CONT", help="имя производной серии")
    ap.add_argument("--root", default="CNYRUBF", help="базовая серия — для спредов и ставки ЦБ")
    ap.add_argument("--tag", default="_w4")
    ap.add_argument("--side", type=int, default=1, choices=[1, -1])
    ap.add_argument("--top-pct", type=float, default=10.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo-fraction", type=float, default=0.01)
    ap.add_argument("--test", action="store_true", help="открыть test (по умолчанию — только train)")
    ap.add_argument("--features", choices=["basis", "carry"], default="basis",
                    help="basis — набор Этапа 1 (нужен спот, есть только у CNYRUBF); "
                         "carry — набор для серий без спота (BR и др.)")
    a = ap.parse_args()
    global MKT
    MKT = MKT if a.features == "basis" else NOBASIS + CARRY

    held, rolls = load_held(a.name)
    tr = pd.read_csv(f"{D}{a.name}_train{a.tag}_v2.csv", parse_dates=["time", "t1"]).dropna(subset=MKT + ["target"])

    if not a.test:
        t_min, t_max = tr["time"].min(), tr["time"].max()
        span = t_max - t_min
        embargo = span * a.embargo_fraction
        edges = [t_min + span * i / (a.folds + 1) for i in range(a.folds + 2)]
        rows = []
        for k in range(1, a.folds + 1):
            lo, hi = edges[k], edges[k + 1]
            te = tr[(tr["time"] >= lo) & (tr["time"] < hi)]
            fit_df = tr[(tr["time"] < lo) & (tr["t1"] < lo) & (tr["time"] < lo - embargo)]
            if len(fit_df) < 2000 or len(te) < 500 or te["target"].nunique() < 2:
                continue
            spread = roll_spreads(a.root, str(te["time"].min().date()), str(te["time"].max().date())).dropna().to_dict()
            m = fit(MKT, fit_df)
            thr = np.percentile(m.predict_proba(fit_df[MKT])[:, 1], 100 - a.top_pct)
            t = build_trades(te, m.predict_proba(te[MKT])[:, 1], thr, held, rolls, spread, a.side)
            if not len(t):
                continue
            n, n1 = net_bp(t), net_bp(t, 1.0)
            rows.append({"окно": k, "период": f"{lo.date()}..{hi.date()}", "сделок": len(t),
                         "валовая": round(t.ret_gross.mean() * 1e4, 2),
                         "издержки": round(t.cost_bp.mean(), 2),
                         "из них ролл": round(t.roll_cost_bp.mean(), 2),
                         "сделок через ролл": int((t.n_rolls > 0).sum()),
                         "чистая": round(n.mean(), 2),
                         "t": round(n.mean() / n.std() * np.sqrt(len(n)), 2),
                         "чистая +1 тик SL": round(n1.mean(), 2)})
        res = pd.DataFrame(rows)
        print(f"=== {a.name}: walk-forward внутри train, test НЕ открыт ===")
        print(res.to_string(index=False))
        print(f"\nсредняя чистая по окнам: {res['чистая'].mean():+.2f} б.п./сделку, "
              f"окон в плюсе {int((res['чистая'] > 0).sum())} из {len(res)}")
        return

    te = pd.read_csv(f"{D}{a.name}_test{a.tag}_v2.csv", parse_dates=["time", "t1"]).dropna(subset=MKT + ["target"]).sort_values("time")
    spread = roll_spreads(a.root, str(te["time"].min().date()), str(te["time"].max().date())).dropna().to_dict()
    print("спред (Roll, круговой) по контрактам, б.п.: " +
          ", ".join(f"{k} {v:.2f}" for k, v in sorted(spread.items(), key=lambda x: x[1])))
    m = fit(MKT, tr)
    thr = np.percentile(m.predict_proba(tr[MKT])[:, 1], 100 - a.top_pct)
    proba = m.predict_proba(te[MKT])[:, 1]
    t = build_trades(te, proba, thr, held, rolls, spread, a.side)

    days = pd.date_range(te["time"].min().tz_localize(None).normalize(),
                         te["time"].max().tz_localize(None).normalize(), freq="D")
    kr_dates, kr_rates = get_rate_series("cbr_key_rate")
    rf_daily = pd.Series([rate_as_of(d.date(), kr_dates, kr_rates) / 100 / 365 for d in days], index=days)
    years = len(days) / 365.25
    rf = stats((1 + rf_daily).cumprod(), years)["годовых"]

    n = net_bp(t)
    eq, in_pos = equity(t, days, rf_daily)
    s = stats(eq, years)
    print(f"\n=== {a.name}: TEST {days[0].date()}..{days[-1].date()} ({years:.2f} года) ===")
    print(f"сделок {len(t)}, дней в позиции {in_pos:.0%}, "
          f"сделок через ролл {(t.n_rolls > 0).sum()} ({(t.n_rolls > 0).mean():.0%})")
    print(f"валовая (по NAV, контанго уже внутри) {t.ret_gross.mean()*1e4:+.2f} б.п./сделку")
    print(f"издержки {t.cost_bp.mean():.2f} б.п. (из них ролл {t.roll_cost_bp.mean():.2f})")
    print(f"ЧИСТАЯ {n.mean():+.2f} б.п./сделку (t={n.mean()/n.std()*np.sqrt(len(n)):.2f}); "
          f"с проскальзыванием +1 тик на SL {net_bp(t, 1.0).mean():+.2f}")
    print(f"итог {s['итог']*100:+.2f}% | {s['годовых']*100:+.2f}% годовых | "
          f"безрисковая {rf*100:.2f}% | maxDD {s['maxDD']*100:.2f}% | Sharpe {s['Sharpe']:.2f}")

    rng = np.random.default_rng(0)
    ctrl = [net_bp(build_trades(te, rng.permutation(proba), thr, held, rolls, spread, a.side)).mean()
            for _ in range(5)]
    print(f"контроль (перемешанные вероятности, 5 прогонов): {np.mean(ctrl):+.2f} б.п. "
          f"(разброс {np.std(ctrl):.2f})")


if __name__ == "__main__":
    main()
