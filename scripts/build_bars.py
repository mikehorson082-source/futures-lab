"""
Этап 1 (CNYRUBF, любая root-серия через --root) — строит dollar bars по
processing/bars.py и печатает приёмочную диагностику.

Приёмочный критерий: если бары нарезаны по обороту, а не по времени,
длительность бара в минутах не должна систематически расти в пятницу или
перед выходными — распределение по дням недели должно быть примерно
ровным. Если пятница/выходные выделяются — граница дня (db/calendar.py)
не сработала как задумано, и это нужно разобрать, а не проигнорировать
(см. CLAUDE.md, п.4).

Только чтение из БД, ничего не пишет и не сохраняет — как
diagnose_roll_splice.py и compute_basis.py, первый проход без порогов
"хорошо/плохо".

Запуск:
    .venv/bin/python -m scripts.build_bars --root CNYRUBF
"""
import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db.calendar import get_trading_day_segments
from db.database import engine
from processing.bars import assign_segments, build_dollar_bars, calibrate_threshold, get_contract_candles

DOW_NAMES = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def get_contracts(root_symbol: str):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT figi, ticker FROM futures_contracts WHERE root_symbol = :root ORDER BY expiration_date"),
            {"root": root_symbol},
        ).fetchall()


def main():
    parser = argparse.ArgumentParser(description="Строит dollar bars по root-серии и печатает приёмочную диагностику.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--target-bars-per-day", type=int, default=30)
    parser.add_argument("--min-candles", type=int, default=50, help="Пропускать контракты с меньшим числом свечей.")
    args = parser.parse_args()

    contracts = get_contracts(args.root)
    segments = get_trading_day_segments(args.root)
    print(f"{args.root}: {len(contracts)} контрактов, {len(segments)} торговых сегментов в календаре.")

    all_bars = []
    per_contract = []
    total_candles = 0

    for c in contracts:
        candles = get_contract_candles(c.figi)
        if len(candles) < args.min_candles:
            continue
        total_candles += len(candles)
        times = [t for t, _, _ in candles]
        seg_ids = assign_segments(times, segments)
        threshold = calibrate_threshold(candles, seg_ids, args.target_bars_per_day)
        if threshold is None:
            continue
        bars = build_dollar_bars(candles, seg_ids, threshold)
        per_contract.append((c.ticker, threshold, len(candles), len(bars)))
        for b in bars:
            b["ticker"] = c.ticker
        all_bars.extend(bars)

    print(
        f"\nВсего баров: {len(all_bars)} из {total_candles} исходных минутных свечей "
        f"({len(all_bars) / max(total_candles, 1) * 100:.2f}% от исходного числа строк)."
    )

    print("\nПорог оборота на контракт (топ-5 самых ликвидных по числу баров):")
    for ticker, thr, n_candles, n_bars in sorted(per_contract, key=lambda x: -x[3])[:5]:
        print(f"  {ticker}: порог={thr:,.0f} руб, {n_candles} свечей -> {n_bars} баров")

    by_dow = defaultdict(list)
    for b in all_bars:
        dow = b["time"].weekday()
        duration_min = (b["time"] - b["open_time"]).total_seconds() / 60.0 + 1
        by_dow[dow].append(duration_min)

    print("\nДлительность бара (мин) по дню недели закрытия:")
    print(f"  {'День':4} {'N':>8} {'mean':>10} {'median':>10} {'p90':>10}")
    for dow in range(7):
        durs = sorted(by_dow.get(dow, []))
        if not durs:
            continue
        p90 = durs[int(len(durs) * 0.9)]
        print(f"  {DOW_NAMES[dow]:4} {len(durs):8} {statistics.mean(durs):10.1f} {statistics.median(durs):10.1f} {p90:10.1f}")


if __name__ == "__main__":
    main()
