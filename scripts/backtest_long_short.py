"""Этап 1 — совместная стратегия лонг+шорт (plan.md, раздел 25).

Лонг и шорт — это ДВЕ РАЗНЫЕ разметки (processing/labeling.py, side=±1) и
две разные модели. Перевернуть вероятность лонговой модели нельзя:
барьеры асимметричны (TP вдвое дальше SL), поэтому исход "SL первым" —
это движение вдвое меньшего размера, а не зеркальный сигнал.

Здесь оба сигнала сводятся в ОДНУ позиционную линию: одна позиция за раз,
хронологически берётся тот сигнал, который пришёл раньше, независимо от
стороны. Издержки — по спреду своего контракта (scripts/backtest.py).

По умолчанию считает walk-forward ВНУТРИ train. Test открывается только с
явным --test (см. раздел 24.3 — каждый взгляд на test размывает гарантию).

Запуск:
    .venv/bin/python -m scripts.backtest_long_short --root CNYRUBF
    .venv/bin/python -m scripts.backtest_long_short --root CNYRUBF --test
"""
import argparse

import numpy as np
import pandas as pd

from scripts.backtest import COMMISSION_BP, equity_curve, roll_spreads, stats
from scripts.compute_basis import get_rate_series, rate_as_of
from scripts.final_eval import MKT, fit

KEYS = ["ticker", "time"]


def candidates(df, proba, thr, side):
    """События-кандидаты одной стороны: вход по открытию СЛЕДУЮЩЕГО бара."""
    d = df.copy()
    d["p"] = proba
    d = d.sort_values(["ticker", "time"])
    d["entry_price"] = d.groupby("ticker", sort=False)["open"].shift(-1)
    d["entry_time"] = d.groupby("ticker", sort=False)["time"].shift(-1)
    d = d.dropna(subset=["entry_price", "entry_time", "exit_price", "t1"])
    d = d[d["p"] >= thr]
    return pd.DataFrame({
        "entry_time": d["entry_time"], "exit_time": d["t1"], "ticker": d["ticker"],
        "reason": d["exit_reason"], "side": side,
        "ret_gross": side * (d["exit_price"] - d["entry_price"]) / d["entry_price"],
    })


def sequence(cands):
    """Одна позиция за раз: хронологически, первый доступный сигнал."""
    c = cands.sort_values("entry_time")
    out, free_at = [], None
    for r in c.itertuples():
        if free_at is not None and r.entry_time < free_at:
            continue
        out.append(r._asdict())
        free_at = r.exit_time
    return pd.DataFrame(out)


def summarize(tag, trades, cost_map, ctx=None):
    if not len(trades):
        print(f"  {tag:22s} сделок нет")
        return
    net = trades["ret_gross"] * 1e4 - trades["ticker"].map(cost_map)
    net = net.dropna()
    t = net.mean() / net.std() * np.sqrt(len(net))
    sides = trades["side"].value_counts()
    print(f"  {tag:22s} сделок {len(net):5d} (лонг {sides.get(1,0):4d} / шорт {sides.get(-1,0):4d})"
          f"  валовая {trades.ret_gross.mean()*1e4:+7.2f}  ЧИСТАЯ {net.mean():+7.2f}  t={t:5.2f}")
    if ctx is not None:
        # прибыль НА СДЕЛКУ не отвечает на вопрос "сколько заработал капитал":
        # у комбинации сделок больше, но каждая в среднем слабее. Нужна кривая.
        days, rf_daily, years, rf_годовых = ctx
        eq, in_pos = equity_curve(trades, days, rf_daily, cost_map)
        st = stats(eq, years)
        print(f"  {'':22s} итог {st['итог']*100:+7.2f}% | {st['годовых']*100:+7.2f}% годовых "
              f"(безрисковая {rf_годовых*100:.2f}%) | maxDD {st['maxDD']*100:6.2f}% | "
              f"Sharpe {st['Sharpe']:5.2f} | дней в позиции {in_pos:.0%}")
    return net.mean()


def load(root, tag, part):
    return pd.read_csv(f"data/features/{root}_{part}{tag}_v2.csv", parse_dates=["time", "t1"]) \
        .dropna(subset=MKT + ["target"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="CNYRUBF")
    ap.add_argument("--long-tag", default="_w4")
    ap.add_argument("--short-tag", default="_w4s")
    ap.add_argument("--top-pct", type=float, default=10.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo-fraction", type=float, default=0.01)
    ap.add_argument("--test", action="store_true", help="открыть test (иначе walk-forward в train)")
    a = ap.parse_args()

    L = load(a.root, a.long_tag, "train")
    S = load(a.root, a.short_tag, "train")

    def run(fit_L, fit_S, ev_L, ev_S, tag):
        # издержки оцениваются в ОКНЕ СДЕЛОК, а не по всей истории контракта:
        # спред контракта в его дальней молодости в разы шире, чем когда он
        # становится ближним, и усреднение по жизни исказило бы издержки.
        cost_map = (roll_spreads(a.root, str(ev_L["time"].min().date()),
                                 str(ev_L["time"].max().date())) + COMMISSION_BP).to_dict()
        mL, mS = fit(MKT, fit_L), fit(MKT, fit_S)
        thrL = np.percentile(mL.predict_proba(fit_L[MKT])[:, 1], 100 - a.top_pct)
        thrS = np.percentile(mS.predict_proba(fit_S[MKT])[:, 1], 100 - a.top_pct)
        cL = candidates(ev_L, mL.predict_proba(ev_L[MKT])[:, 1], thrL, 1)
        cS = candidates(ev_S, mS.predict_proba(ev_S[MKT])[:, 1], thrS, -1)
        print(f"\n{tag}")
        ctx = None
        if a.test:
            days = pd.date_range(ev_L["time"].min().tz_localize(None).normalize(),
                                 ev_L["time"].max().tz_localize(None).normalize(), freq="D")
            kr_dates, kr_rates = get_rate_series("cbr_key_rate")
            rf_daily = pd.Series([rate_as_of(d.date(), kr_dates, kr_rates) / 100 / 365
                                  for d in days], index=days)
            years = len(days) / 365.25
            ctx = (days, rf_daily, years, stats((1 + rf_daily).cumprod(), years)["годовых"])
        r = {}
        r["лонг"] = summarize("только лонг", sequence(cL), cost_map, ctx)
        r["шорт"] = summarize("только шорт", sequence(cS), cost_map, ctx)
        r["оба"] = summarize("лонг+шорт", sequence(pd.concat([cL, cS])), cost_map, ctx)
        return r

    if a.test:
        print("⚠️  ОТКРЫВАЕТСЯ TEST (см. plan.md, раздел 24.3)")
        run(L, S, load(a.root, a.long_tag, "test"), load(a.root, a.short_tag, "test"), "=== TEST ===")
        return

    t_min, t_max = L["time"].min(), L["time"].max()
    span = t_max - t_min
    emb = span * a.embargo_fraction
    edges = [t_min + span * i / (a.folds + 1) for i in range(a.folds + 2)]
    acc = []
    for k in range(1, a.folds + 1):
        lo, hi = edges[k], edges[k + 1]
        sub = lambda d: (d[(d.time >= lo) & (d.time < hi)],
                         d[(d.time < lo) & (d.t1 < lo) & (d.time < lo - emb)])
        evL, ftL = sub(L)
        evS, ftS = sub(S)
        if len(ftL) < 2000 or len(evL) < 500:
            continue
        acc.append(run(ftL, ftS, evL, evS, f"окно {k}: {lo.date()}..{hi.date()}"))

    df = pd.DataFrame(acc)
    print("\n=== сводка по окнам (чистая б.п./сделку) ===")
    print(pd.DataFrame({"среднее": df.mean(), "окон в плюсе": (df > 0).sum(),
                        "худшее окно": df.min()}).round(2).to_string())


if __name__ == "__main__":
    main()
