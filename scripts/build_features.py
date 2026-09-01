"""
Этап 1 (CNYRUBF, любая root-серия через --root) — полный пайплайн
свечи -> dollar bars (processing/bars.py) -> признаки (processing/features.py),
результат сохраняется в data/features/<root>_bars_features.csv.

В отличие от scripts/build_bars.py, scripts/compute_basis.py и
scripts/diagnose_roll_splice.py (только диагностика, ничего не пишут) —
это первый шаг, материализующий результат на диск: дальше признаки нужны
как вход для разметки/модели, пересчитывать их из сырых свечей на каждый
запуск было бы расточительно. data/ и *.csv не попадают в git (.gitignore).

Запуск:
    .venv/bin/python -m scripts.build_features --root CNYRUBF
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db.calendar import get_trading_day_segments
from db.database import engine
from processing.bars import (
    DEFAULT_DTE_BUCKETS,
    assign_segments,
    build_dollar_bars,
    calibrate_threshold_curve,
    get_contract_candles,
    thresholds_for_contract,
)
from processing.features import (
    FEATURE_COLUMNS,
    add_basis_features,
    add_price_features,
    add_structural_features,
    build_daily_volume,
    build_rank_and_share,
    get_contracts_meta,
    load_basis_inputs,
)
from processing.splitting import compute_split_time, get_root_time_range


def get_contracts(root_symbol: str):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT figi, ticker FROM futures_contracts WHERE root_symbol = :root ORDER BY expiration_date"),
            {"root": root_symbol},
        ).fetchall()


def main():
    parser = argparse.ArgumentParser(description="Строит dollar bars + признаки для root-серии, сохраняет в data/features/.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--target-bars-per-day", type=int, default=30)
    parser.add_argument("--min-candles", type=int, default=50)
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="ДОЛЖНО совпадать с --test-fraction в scripts/build_split.py — иначе порог "
             "dollar bars и реальный train/test разрез будут calibровать/резать по разным границам.",
    )
    args = parser.parse_args()
    root = args.root

    contracts = get_contracts(root)
    segments = get_trading_day_segments(root)
    contracts_meta = get_contracts_meta(root)
    meta_by_ticker = {m["ticker"]: m for m in contracts_meta}

    t_min, t_max = get_root_time_range(root)
    split_time = compute_split_time(t_min, t_max, args.test_fraction)
    print(f"{root}: {len(contracts)} контрактов, {len(segments)} торговых сегментов в календаре.")
    print(f"Граница train/test (для калибровки порога): {split_time} "
          f"(test_fraction={args.test_fraction}, ДОЛЖНА совпадать с scripts.build_split).")

    # 1. Бары на каждый контракт отдельно (processing/bars.py). Порог
    # калибруется ТОЛЬКО по train-периоду (до split_time) — раздел 10.1/12 plan.md.
    candles_by_ticker, seg_ids_by_ticker = {}, {}
    for c in contracts:
        candles = get_contract_candles(c.figi)
        if len(candles) < args.min_candles:
            continue
        times = [t for t, _, _ in candles]
        candles_by_ticker[c.ticker] = candles
        seg_ids_by_ticker[c.ticker] = assign_segments(times, segments)

    expiration_by_ticker = {m["ticker"]: m["expiration_date"] for m in contracts_meta}
    bucket_thresholds = calibrate_threshold_curve(
        candles_by_ticker, seg_ids_by_ticker, expiration_by_ticker,
        args.target_bars_per_day, before=split_time,
    )
    bucket_labels = [f"<= {DEFAULT_DTE_BUCKETS[0]}д"] + [
        f"{DEFAULT_DTE_BUCKETS[i-1]}-{DEFAULT_DTE_BUCKETS[i]}д" for i in range(1, len(DEFAULT_DTE_BUCKETS))
    ] + [f"> {DEFAULT_DTE_BUCKETS[-1]}д"]
    print("Порог dollar bars по бакетам days_to_expiration (train-период, пул всех контрактов):")
    for label, thr in zip(bucket_labels, bucket_thresholds):
        print(f"  {label:<10} {thr:,.0f} руб" if thr is not None else f"  {label:<10} (нет train-данных вообще)")

    bars_by_ticker = {}
    for ticker, candles in candles_by_ticker.items():
        exp = expiration_by_ticker[ticker]
        per_candle_thresholds = thresholds_for_contract(candles, exp, bucket_thresholds)
        if any(t is None for t in per_candle_thresholds):
            continue  # ни одного train-дня ни у кого в этом бакете — контракт пропущен
        bars_by_ticker[ticker] = build_dollar_bars(candles, seg_ids_by_ticker[ticker], per_candle_thresholds)
    total_bars = sum(len(b) for b in bars_by_ticker.values())
    print(f"Баров построено: {total_bars} по {len(bars_by_ticker)} контрактам.")

    # 2. Структурные признаки (нужен общий по root дневной оборот -> ранг/доля)
    daily_volume = build_daily_volume(bars_by_ticker)
    rank_share = build_rank_and_share(contracts_meta, daily_volume)
    for ticker, bars in bars_by_ticker.items():
        add_structural_features(bars, ticker, meta_by_ticker[ticker], rank_share)
    print("Структурные признаки добавлены (days_to_expiration, life_fraction_remaining, contract_rank, volume_share).")

    # 3. Базис (только если для root есть спот-пара, см. ROOT_TO_SPOT_PAIR)
    basis_inputs = load_basis_inputs(root)
    if basis_inputs is not None:
        spot_map, kr_dates, kr_rates, fr_dates, fr_rates = basis_inputs
        for ticker, bars in bars_by_ticker.items():
            add_basis_features(bars, meta_by_ticker[ticker], root, spot_map, kr_dates, kr_rates, fr_dates, fr_rates)
        print("Базис добавлен (basis_pct_rub_only, basis_pct_full).")
    else:
        print(f"Для {root} нет спот-пары в ROOT_TO_SPOT_PAIR (scripts/compute_basis.py) — базис пропущен.")

    # 4. Ценовые признаки — строго внутри контракта, окно не переходит через ролл
    for bars in bars_by_ticker.values():
        add_price_features(bars)
    print("Ценовые признаки добавлены (log_return_1, volatility_20, momentum_10).")

    # 5. Сборка и сохранение
    all_rows = []
    for ticker, bars in bars_by_ticker.items():
        for b in bars:
            row = dict(b)
            row["ticker"] = ticker
            row["root_symbol"] = root
            all_rows.append(row)
    all_rows.sort(key=lambda r: r["time"])

    out_dir = Path(__file__).resolve().parent.parent / "data" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{root}_bars_features.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in FEATURE_COLUMNS})
    print(f"\nСохранено: {out_path} ({len(all_rows)} строк).")

    # Сводка по NaN/None на колонку — честно показать, где признак не покрывает весь ряд
    print("\nПокрытие признаков (доля непустых значений):")
    for col in FEATURE_COLUMNS:
        non_null = sum(1 for r in all_rows if r.get(col) is not None)
        print(f"  {col:<24} {non_null / max(len(all_rows), 1) * 100:5.1f}%")

    numeric_cols = ["life_fraction_remaining", "volume_share", "basis_pct_rub_only", "log_return_1", "volatility_20", "momentum_10"]
    print("\nБазовая статистика по числовым признакам (без None):")
    for col in numeric_cols:
        vals = [r[col] for r in all_rows if r.get(col) is not None]
        if not vals:
            continue
        print(f"  {col:<24} mean {statistics.mean(vals):+8.4f}  std {statistics.pstdev(vals):8.4f}  min {min(vals):+8.4f}  max {max(vals):+8.4f}")


if __name__ == "__main__":
    main()
