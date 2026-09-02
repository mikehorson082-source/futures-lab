"""
Строит НЕПРЕРЫВНЫЙ ряд по одной root-серии методом ETF Trick
(processing/etf_trick.py, де Прадо §2.4.1) и печатает диагностику.

Это альтернатива pool-архитектуре Этапа 1 (каждый контракт отдельно):
там ценовые окна обрывались на каждом ролле, здесь история одна и
непрерывная. Что из этого лучше для модели — отдельный эксперимент
(scripts/build_continuous_dataset.py), этот скрипт только строит ряд и
показывает, что именно склейка сделала с данными.

Результат: data/features/<root>_continuous_1m.csv
    time, ticker, raw_close, nav, volume, dollar_volume, is_roll

Запуск:
    .venv/bin/python -m scripts.build_continuous_series --root CNYRUBF
    .venv/bin/python -m scripts.build_continuous_series --root CNYRUBF \
        --roll-rule days --days-before 5
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db.database import engine
from processing.bars import get_contract_candles
from processing.etf_trick import (
    apply_etf_trick,
    build_roll_schedule_by_days,
    build_roll_schedule_by_volume,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "features"


def get_contracts(root):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT figi, ticker, expiration_date, first_trade_date
            FROM futures_contracts WHERE root_symbol = :root ORDER BY expiration_date
        """), {"root": root}).fetchall()
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Непрерывный ряд по ETF Trick.")
    p.add_argument("--root", default="CNYRUBF")
    p.add_argument("--roll-rule", choices=["volume", "days"], default="volume",
                   help="volume — по перетоку ликвидности (причинно, со следующего дня); "
                        "days — фиксированно за N дней до экспирации")
    p.add_argument("--days-before", type=int, default=5)
    p.add_argument("--roll-cost-bps", type=float, default=0.0,
                   help="издержки одного ролла в б.п. NAV (закрыть старый + открыть новый)")
    p.add_argument("--min-candles", type=int, default=50)
    p.add_argument("--name", default=None,
                   help="имя производной серии (по умолчанию <root>_CONT). Разные "
                        "правила ролла кладутся в разные файлы: CNYRUBF_CONT5 и т.п.")
    a = p.parse_args(argv)

    contracts = [c for c in get_contracts(a.root)]
    candles, daily_vol, expiration, first_d, last_d = {}, {}, {}, {}, {}
    for c in contracts:
        cs = get_contract_candles(c.figi)
        if len(cs) < a.min_candles:
            continue
        candles[c.ticker] = cs
        expiration[c.ticker] = c.expiration_date
        first_d[c.ticker] = cs[0][0].date()
        last_d[c.ticker] = cs[-1][0].date()
        dv = defaultdict(float)
        for t, close, vol in cs:
            dv[t.date()] += close * vol
        daily_vol[c.ticker] = dict(dv)
    order = sorted(candles, key=lambda t: expiration[t])
    print(f"{a.root}: контрактов с данными {len(order)}, "
          f"{min(first_d.values())} … {max(last_d.values())}")

    if a.roll_rule == "volume":
        schedule = build_roll_schedule_by_volume(daily_vol, order, expiration, last_d)
    else:
        schedule = build_roll_schedule_by_days(order, expiration, first_d, a.days_before)

    series, rolls = apply_etf_trick(candles, schedule, roll_cost_bps=a.roll_cost_bps)
    print(f"правило ролла: {a.roll_rule}, переходов: {len(rolls)}, "
          f"минутных свечей в непрерывном ряду: {len(series)}\n")

    print("Разрывы на роллах — то, что ETF Trick НЕ пропустил в ряд:")
    print(f"{'дата':12s} {'из':>12s} {'в':>12s} {'цена до':>9s} {'цена после':>11s} {'разрыв %':>9s}")
    for r in rolls:
        print(f"{str(r['date']):12s} {r['from']:>12s} {r['to']:>12s} "
              f"{r['price_from']:9.3f} {r['price_to']:11.3f} {r['gap_pct']:+9.2f}")
    if rolls:
        gaps = [abs(r["gap_pct"]) for r in rolls]
        naive_extra = 1.0
        for r in rolls:
            naive_extra *= (1.0 + r["gap_pct"] / 100.0)
        print(f"\nсредний |разрыв| {sum(gaps)/len(gaps):.2f}%, максимум {max(gaps):.2f}%")
        print(f"Наивная склейка встык приписала бы ряду доходность "
              f"{(naive_extra - 1) * 100:+.1f}% из воздуха — ровно произведение "
              f"этих разрывов; в NAV её нет.")

    nav0, nav1 = series[0]["nav"], series[-1]["nav"]
    print(f"\nNAV: {nav0:.4f} -> {nav1:.4f}  ({(nav1/nav0 - 1)*100:+.1f}% за всю историю, "
          f"издержки ролла {a.roll_cost_bps:.0f} б.п.)")
    print(f"Цена ближнего контракта в начале/в конце: "
          f"{series[0]['raw_close']:.3f} -> {series[-1]['raw_close']:.3f}")

    out = DATA / f"{a.name or (a.root + '_CONT')}_continuous_1m.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "ticker", "raw_close", "nav", "volume", "dollar_volume", "is_roll"])
        w.writeheader()
        for row in series:
            w.writerow(row)
    print(f"\nСохранено: {out} ({len(series)} строк)")


if __name__ == "__main__":
    main()
