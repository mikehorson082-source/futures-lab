"""Этап 1 — бэктест с издержками на test-периоде (CLAUDE.md, best practices).

Модель — финальная конфигурация раздела 22 (6 стационарных рыночных
признаков + логрегрессия), обучена на train, применяется к test.

Что здесь сделано честно, и почему без этого бэктест был бы фикцией:

1. ОДНА ПОЗИЦИЯ ЗА РАЗ. Окна меток перекрываются (mean overlap 7.8,
   раздел 11) — взять все сигналы значило бы держать ~8 позиций
   одновременно на один капитал.
2. ВХОД ПО ОТКРЫТИЮ СЛЕДУЮЩЕГО БАРА (сигнал известен на закрытии — войти
   по нему же нельзя, это заглядывание на бар вперёд).
3. ИЗДЕРЖКИ ПО СПРЕДУ СВОЕГО КОНТРАКТА, не единой ставкой. Спред
   оценивается по Roll (2*sqrt(-cov соседних доходностей)) на МИНУТНЫХ
   свечах каждого контракта. Разброс между контрактами — в 40 раз
   (1.3 б.п. на ближнем против 54 б.п. на дальнем), единая ставка
   полностью искажает картину.
4. ФИЛЬТР ЛИКВИДНОСТИ. В реальности контракт со спредом 54 б.п. никто не
   торгует. Порог задаётся по ЛИКВИДНОСТИ (volume_share — доля дневного
   оборота серии, известна в реальном времени), не по доходности.
5. СРАВНЕНИЕ С БЕЗРИСКОВОЙ СТАВКОЙ: капитал вне позиции лежит под
   ключевую ставку ЦБ.
6. КОНТРОЛЬ НА ШУМ: та же механика на перемешанных вероятностях.

Запуск:
    .venv/bin/python -m scripts.backtest --root CNYRUBF
"""
import argparse

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.database import engine
from scripts.compute_basis import get_rate_series, rate_as_of
from scripts.final_eval import MKT, fit

COMMISSION_BP = 0.5  # биржевая+брокерская комиссия, круговая, оценка


def roll_spreads(root: str, since: str, until: str = None) -> pd.Series:
    """Эффективный круговой спред по контрактам, оценка Roll на минутных свечах.

    Окно [since, until) должно совпадать с периодом, в котором происходят
    сделки: спред контракта меняется по ходу его жизни (в молодости, пока
    он дальний, он в разы шире, чем когда становится ближним). Оценка по
    всей истории контракта завысила бы издержки его активного периода.

    Roll предполагает тиковые сделки; на минутных свечах оценка смещена
    ВВЕРХ (в отрицательную автокорреляцию попадает и настоящий возврат к
    среднему, не только скачки bid-ask). Это верхняя граница, не точное
    значение — и она используется именно как консервативная оценка.
    """
    q = """SELECT c.ticker, fc.time, fc.close FROM futures_candles fc
           JOIN futures_contracts c ON c.figi = fc.figi
           WHERE c.root_symbol = :root AND fc.time >= :since
             AND (:until IS NULL OR fc.time < CAST(:until AS timestamptz))
           ORDER BY c.ticker, fc.time"""
    d = pd.read_sql(text(q), engine, params={"root": root, "since": since, "until": until})
    out = {}
    for t, g in d.groupby("ticker"):
        r = np.diff(np.log(g["close"].astype(float).values))
        if len(r) < 2000:
            continue
        cov = np.cov(r[1:], r[:-1])[0, 1]
        out[t] = 2 * np.sqrt(-cov) * 1e4 if cov < 0 else np.nan
    return pd.Series(out, name="spread_bp")


def build_trades(te, proba, thr, min_volume_share=0.0, side=1):
    """Хронологический отбор сигналов с правилом одной позиции за раз.

    side=+1 лонг, -1 шорт: у шорта прибыль даёт ПАДЕНИЕ цены, поэтому
    доходность считается зеркально (иначе знак был бы перевёрнут).
    """
    d = te.copy()
    d["p"] = proba
    d = d.sort_values(["ticker", "time"])
    d["entry_price"] = d.groupby("ticker", sort=False)["open"].shift(-1)
    d["entry_time"] = d.groupby("ticker", sort=False)["time"].shift(-1)
    d = d.dropna(subset=["entry_price", "entry_time", "exit_price", "t1"])
    d = d[(d["p"] >= thr) & (d["volume_share"] >= min_volume_share)].sort_values("entry_time")

    trades, free_at = [], None
    for r in d.itertuples():
        if free_at is not None and r.entry_time < free_at:
            continue
        trades.append({
            "entry_time": r.entry_time, "exit_time": r.t1, "ticker": r.ticker,
            "reason": r.exit_reason,
            "ret_gross": side * (r.exit_price - r.entry_price) / r.entry_price,
        })
        free_at = r.t1
    return pd.DataFrame(trades)


def equity_curve(trades, days, rf_daily, cost_bp_by_ticker):
    """Капитал: в сделке — её чистый результат, вне сделки — безрисковая ставка."""
    if not len(trades):
        return (1 + rf_daily).cumprod(), 0.0
    cost = trades["ticker"].map(cost_bp_by_ticker).fillna(0) / 1e4
    net = (trades["ret_gross"] - cost).values
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


def stats(eq, years):
    ret = eq.iloc[-1] - 1
    daily = eq.pct_change().dropna()
    return {"итог": ret, "годовых": (1 + ret) ** (1 / years) - 1,
            "maxDD": (eq / eq.cummax() - 1).min(),
            "Sharpe": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan}


def report(tag, trades, days, rf_daily, cost_map, years, rf_годовых):
    if not len(trades):
        print(f"\n--- {tag}: сделок нет ---")
        return
    cost = trades["ticker"].map(cost_map).fillna(0)
    net = trades["ret_gross"] * 1e4 - cost
    eq, in_pos = equity_curve(trades, days, rf_daily, cost_map)
    s = stats(eq, years)
    print(f"\n--- {tag} ---")
    print(f"  сделок {len(trades)}, контрактов {trades.ticker.nunique()}, "
          f"дней в позиции {in_pos:.0%}")
    print(f"  валовая {trades.ret_gross.mean()*1e4:+.2f} б.п./сделку "
          f"(t={trades.ret_gross.mean()/trades.ret_gross.std()*np.sqrt(len(trades)):.2f})")
    print(f"  ЧИСТАЯ  {net.mean():+.2f} б.п./сделку "
          f"(t={net.mean()/net.std()*np.sqrt(len(net)):.2f})")
    print(f"  итог {s['итог']*100:+.2f}% за период | {s['годовых']*100:+.2f}% годовых | "
          f"безрисковая {rf_годовых*100:.2f}% | maxDD {s['maxDD']*100:.2f}% | Sharpe {s['Sharpe']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="CNYRUBF")
    ap.add_argument("--tag", default="", help="вариант ширины барьеров, см. build_labels --tag")
    ap.add_argument("--side", type=int, default=1, choices=[1, -1],
                    help="сторона сделки; должна совпадать со стороной разметки")
    ap.add_argument("--top-pct", type=float, default=10.0)
    ap.add_argument("--min-volume-share", type=float, default=0.0,
                    help="фильтр ликвидности; по умолчанию ВЫКЛЮЧЕН — walk-forward "
                         "внутри train показал, что он ухудшает результат (раздел 24)")
    a = ap.parse_args()

    tr = pd.read_csv(f"data/features/{a.root}_train{a.tag}_v2.csv", parse_dates=["time"])
    te = pd.read_csv(f"data/features/{a.root}_test{a.tag}_v2.csv", parse_dates=["time", "t1"])
    te = te.dropna(subset=MKT + ["target"]).sort_values("time")

    model = fit(MKT, tr)
    proba = model.predict_proba(te[MKT])[:, 1]
    trf = tr.dropna(subset=MKT + ["target"])
    thr = np.percentile(model.predict_proba(trf[MKT])[:, 1], 100 - a.top_pct)

    # окно оценки спреда = окно сделок (см. docstring roll_spreads)
    sp = roll_spreads(a.root, str(te["time"].min().date()), str(te["time"].max().date()))
    cost_map = (sp + COMMISSION_BP).to_dict()
    print("оценка круговых издержек по контрактам (Roll + комиссия), б.п.:")
    print("  " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(cost_map.items(), key=lambda x: x[1])))

    days = pd.date_range(te["time"].min().tz_localize(None).normalize(),
                         te["time"].max().tz_localize(None).normalize(), freq="D")
    kr_dates, kr_rates = get_rate_series("cbr_key_rate")
    rf_daily = pd.Series([rate_as_of(d.date(), kr_dates, kr_rates) / 100 / 365 for d in days],
                         index=days)
    years = len(days) / 365.25
    rf_годовых = stats((1 + rf_daily).cumprod(), years)["годовых"]
    print(f"\nпериод test: {days[0].date()} .. {days[-1].date()} ({years:.2f} года), "
          f"безрисковая (ключевая ставка ЦБ) {rf_годовых*100:.2f}% годовых")

    all_tr = build_trades(te, proba, thr, 0.0, a.side)
    report("БЕЗ фильтра ликвидности (все контракты)", all_tr, days, rf_daily, cost_map, years, rf_годовых)

    liq_tr = build_trades(te, proba, thr, a.min_volume_share, a.side)
    report(f"С фильтром ликвидности (volume_share >= {a.min_volume_share})",
           liq_tr, days, rf_daily, cost_map, years, rf_годовых)

    print("\nразбивка по контрактам (с фильтром ликвидности):")
    g = liq_tr.groupby("ticker").agg(сделок=("ret_gross", "size"),
                                     валовая=("ret_gross", lambda s: round(s.mean() * 1e4, 2)))
    g["издержки"] = g.index.map(cost_map).round(2)
    g["чистая"] = (g["валовая"] - g["издержки"]).round(2)
    print(g.to_string())

    rng = np.random.default_rng(0)
    ctrl = []
    for _ in range(5):
        t = build_trades(te, rng.permutation(proba), thr, a.min_volume_share, a.side)
        c = t["ticker"].map(cost_map).fillna(0)
        ctrl.append((t["ret_gross"] * 1e4 - c).mean())
    print(f"\nконтроль (перемешанные вероятности, 5 прогонов): чистая "
          f"{np.mean(ctrl):+.2f} б.п./сделку (разброс {np.std(ctrl):.2f})")


if __name__ == "__main__":
    main()
