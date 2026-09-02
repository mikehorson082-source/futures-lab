"""
Бары + признаки поверх НЕПРЕРЫВНОГО ряда (ETF Trick), т.е. альтернатива
scripts/build_features.py, где каждый контракт жил отдельно.

Что меняется по сравнению с pool-архитектурой Этапа 1:
  * ценовые окна (volatility_20, momentum_10 и производные z-признаки)
    больше НЕ обрываются на каждом ролле — история одна;
  * один бар = один момент рынка, а не N параллельных строк по N живым
    контрактам (в пуле соседние контракты дают почти дублирующие строки);
  * цена бара — NAV из ETF Trick, т.е. то, что реально заработала бы
    позиция «держать ближний контракт и роллировать», вместе с потерей
    на контанго. Разметка Triple Barrier поверх NAV поэтому измеряет
    настоящий P&L стратегии, а не движение котировки, которого не получить.

Базис считается по СЫРОЙ цене удерживаемого контракта (raw_close), а не по
NAV — базис это «фьючерс против спота», NAV тут не при чём.

На выходе — файл в схеме FEATURE_COLUMNS с одним «контрактом»
<root>_CONT, поэтому дальше работают ОБЫЧНЫЕ скрипты пайплайна:
    build_labels --root CNYRUBF_CONT --tag _w4 ...
    build_split  --root CNYRUBF_CONT --split-root CNYRUBF --tag _w4
    build_derived_features --root CNYRUBF_CONT --tag _w4

Запуск:
    .venv/bin/python -m scripts.build_continuous_series  --root CNYRUBF
    .venv/bin/python -m scripts.build_continuous_features --root CNYRUBF
"""
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db.calendar import get_trading_day_segments
from db.database import engine
from processing.bars import assign_segments
from processing.features import (
    FEATURE_COLUMNS,
    add_basis_features,
    add_price_features,
    get_contracts_meta,
    load_basis_inputs,
)
from processing.splitting import compute_split_time, get_root_time_range

DATA = Path(__file__).resolve().parent.parent / "data" / "features"


def read_continuous(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "time": datetime.fromisoformat(r["time"]),
                "ticker": r["ticker"],
                "raw_close": float(r["raw_close"]),
                "nav": float(r["nav"]),
                "volume": int(r["volume"]),
                "dollar_volume": float(r["dollar_volume"]),
            })
    return rows


def all_contracts_daily_volume(root):
    """Дневной оборот КАЖДОГО контракта серии — для volume_share ближнего."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.ticker, fc.time::date AS d, SUM(fc.close * fc.volume) AS dv
            FROM futures_candles fc JOIN futures_contracts c ON c.figi = fc.figi
            WHERE c.root_symbol = :root GROUP BY 1, 2
        """), {"root": root}).fetchall()
    out = defaultdict(dict)
    for r in rows:
        out[r.ticker][r.d] = float(r.dv)
    return out


def add_carry_feature(bars, root, meta_by_ticker, max_stale_hours=24.0):
    """carry_annual — «базис по кривой»: годовая стоимость времени между
    ближним и следующим контрактом.

        carry = ln(P_след / P_ближ) * 365 / (T_след - T_ближ)   [% годовых]

    Зачем это нужно. Базис Этапа 1 (basis_pct_full) считается как «фьючерс
    против спота с поправкой на ставки» и требует спот-курс и две ставки —
    внешние данные, которые есть только для CNYRUBF (ROOT_TO_SPOT_PAIR).
    Для нефти спота на MOEX нет вообще. Календарный спред даёт ту же
    величину — цену времени — но ИЗ САМОЙ КРИВОЙ, без внешних источников,
    поэтому считается для любой серии.

    Цена следующего контракта берётся по последней его свече НЕ ПОЗЖЕ
    времени бара (никакого заглядывания вперёд). Если последняя сделка по
    нему старше max_stale_hours — carry не считается (None): дальний
    контракт бывает неликвиден, и застоявшаяся цена дала бы фиктивный
    спред.
    """
    import numpy as np

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.ticker, fc.time, fc.close FROM futures_candles fc
            JOIN futures_contracts c ON c.figi = fc.figi
            WHERE c.root_symbol = :root ORDER BY c.ticker, fc.time
        """), {"root": root}).fetchall()
    series = defaultdict(list)
    for r in rows:
        series[r.ticker].append((r.time, float(r.close)))
    arrays = {tk: (np.array([t.replace(tzinfo=None) for t, _ in v], dtype="datetime64[ns]"),
                   np.array([c for _, c in v]))
              for tk, v in series.items()}

    order = sorted([tk for tk in arrays if tk in meta_by_ticker],
                   key=lambda tk: meta_by_ticker[tk]["expiration_date"])
    next_of = {tk: order[i + 1] for i, tk in enumerate(order[:-1])}
    stale = np.timedelta64(int(max_stale_hours * 3600), "s")

    n_ok = 0
    for b in bars:
        front = b["held_ticker"]
        nxt = next_of.get(front)
        b["carry_annual"] = None
        if nxt is None or nxt not in arrays:
            continue
        t = np.datetime64(b["time"].replace(tzinfo=None))
        times, closes = arrays[nxt]
        i = np.searchsorted(times, t, side="right") - 1
        if i < 0 or (t - times[i]) > stale:
            continue
        dt_days = (meta_by_ticker[nxt]["expiration_date"] - meta_by_ticker[front]["expiration_date"]).days
        if dt_days <= 0 or b["raw_close"] <= 0 or closes[i] <= 0:
            continue
        b["carry_annual"] = math.log(closes[i] / b["raw_close"]) * 365.0 / dt_days * 100.0
        n_ok += 1
    print(f"carry_annual (календарный спред) посчитан на {n_ok/max(len(bars),1):.1%} баров")


def build_bars(rows, seg_ids, threshold):
    """Dollar bars по непрерывному ряду. Порог — по РЕАЛЬНОМУ рублёвому
    обороту (raw_close * volume), OHLC — по NAV. Бар принудительно
    закрывается на границе торгового сегмента (как в processing/bars.py)."""
    bars, cur, cum, cur_seg = [], None, 0.0, None
    for r, seg in zip(rows, seg_ids):
        if cur is None or seg != cur_seg or cum >= threshold:
            if cur is not None:
                bars.append(cur)
            cur = {"open_time": r["time"], "time": r["time"],
                   "open": r["nav"], "high": r["nav"], "low": r["nav"], "close": r["nav"],
                   "raw_close": r["raw_close"], "volume": 0, "dollar_volume": 0.0,
                   "n_ticks": 0, "held_ticker": r["ticker"]}
            cum = 0.0
            cur_seg = seg
        cur["high"] = max(cur["high"], r["nav"])
        cur["low"] = min(cur["low"], r["nav"])
        cur["close"] = r["nav"]
        cur["raw_close"] = r["raw_close"]
        cur["held_ticker"] = r["ticker"]
        cur["volume"] += r["volume"]
        cur["dollar_volume"] += r["dollar_volume"]
        cur["n_ticks"] += 1
        cur["time"] = r["time"]
        cum += r["dollar_volume"]
    if cur is not None:
        bars.append(cur)
    return bars


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="CNYRUBF")
    p.add_argument("--target-bars-per-day", type=int, default=30)
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="граница train — порог dollar bars калибруется ТОЛЬКО по train")
    p.add_argument("--name", default=None, help="имя производной серии (по умолчанию <root>_CONT)")
    a = p.parse_args(argv)
    cont_name = a.name or f"{a.root}_CONT"

    rows = read_continuous(DATA / f"{cont_name}_continuous_1m.csv")
    segments = get_trading_day_segments(a.root)
    seg_ids = assign_segments([r["time"] for r in rows], segments)

    t_min, t_max = get_root_time_range(a.root)
    split_time = compute_split_time(t_min, t_max, a.test_fraction)

    # порог: средний дневной оборот непрерывного ряда на train / баров в день
    dv_by_day = defaultdict(float)
    for r in rows:
        if r["time"] < split_time:
            dv_by_day[r["time"].date()] += r["dollar_volume"]
    threshold = statistics.mean(dv_by_day.values()) / a.target_bars_per_day
    print(f"{cont_name}: {len(rows)} минутных строк, граница train {split_time.date()}, "
          f"порог dollar bars {threshold:,.0f} руб ({a.target_bars_per_day} баров/день на train)")

    bars = build_bars(rows, seg_ids, threshold)
    print(f"Баров построено: {len(bars)}")

    # --- структурные признаки (по УДЕРЖИВАЕМОМУ контракту)
    meta_by_ticker = {m["ticker"]: m for m in get_contracts_meta(a.root)}
    daily_vol = all_contracts_daily_volume(a.root)
    live_by_date = defaultdict(list)
    for tk, m in meta_by_ticker.items():
        for d in daily_vol.get(tk, {}):
            live_by_date[d].append(tk)
    for b in bars:
        m = meta_by_ticker[b["held_ticker"]]
        d = b["time"].date()
        exp, first = m["expiration_date"], m["first_trade_date"] or m["expiration_date"]
        b["days_to_expiration"] = (exp - d).days
        b["life_fraction_remaining"] = max(0.0, min(1.0, (exp - d).days / max((exp - first).days, 1)))
        b["contract_rank"] = 1  # по построению держим ближний по ликвидности
        total = sum(daily_vol[tk].get(d, 0.0) for tk in live_by_date.get(d, []))
        own = daily_vol[b["held_ticker"]].get(d, 0.0)
        b["volume_share"] = own / total if total > 0 else None

    # --- базис: по СЫРОЙ цене контракта, группами по удерживаемому контракту
    basis_inputs = load_basis_inputs(a.root)
    if basis_inputs is not None:
        spot_map, kr_dates, kr_rates, fr_dates, fr_rates = basis_inputs
        by_held = defaultdict(list)
        for b in bars:
            by_held[b["held_ticker"]].append(b)
        for tk, group in by_held.items():
            proxy = [{"time": b["time"], "close": b["raw_close"]} for b in group]
            add_basis_features(proxy, meta_by_ticker[tk], a.root, spot_map, kr_dates, kr_rates, fr_dates, fr_rates)
            for b, pb in zip(group, proxy):
                b["basis_pct_rub_only"] = pb["basis_pct_rub_only"]
                b["basis_pct_full"] = pb["basis_pct_full"]
        print("Базис добавлен по сырой цене ближнего контракта.")

    add_carry_feature(bars, a.root, meta_by_ticker)

    # --- ценовые признаки: ОДИН проход по всей истории, окна не рвутся на роллах
    add_price_features(bars)
    print("Ценовые признаки добавлены по непрерывному NAV (окна не обрываются на роллах).")

    for b in bars:
        b["ticker"] = cont_name
        b["root_symbol"] = cont_name
    out = DATA / f"{cont_name}_bars_features.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        w.writeheader()
        for b in bars:
            w.writerow({k: b.get(k) for k in FEATURE_COLUMNS})
    print(f"\nСохранено: {out} ({len(bars)} строк)")

    # Сайдкар: какой РЕАЛЬНЫЙ контракт удерживался на каждом баре. В схему
    # FEATURE_COLUMNS он не влезает (её читают общие скрипты пайплайна), а
    # бэктесту он нужен: спред и издержки ролла считаются по конкретному
    # контракту, а не по имени производной серии.
    held_path = DATA / f"{cont_name}_held.csv"
    with open(held_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "held_ticker", "raw_close"])
        w.writeheader()
        for b in bars:
            w.writerow({"time": b["time"], "held_ticker": b["held_ticker"], "raw_close": b["raw_close"]})
    n_rolls = sum(1 for i in range(1, len(bars)) if bars[i]["held_ticker"] != bars[i-1]["held_ticker"])
    print(f"Сохранено: {held_path} (удерживаемый контракт по барам, переходов {n_rolls})")
    print("\nПокрытие признаков:")
    for col in FEATURE_COLUMNS:
        nn = sum(1 for b in bars if b.get(col) is not None)
        print(f"  {col:<24} {nn / max(len(bars),1)*100:5.1f}%")


if __name__ == "__main__":
    main()
