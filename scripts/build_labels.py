"""
Этап 1 — размечает уже построенные бары+признаки
(data/features/<root>_bars_features.csv, см. scripts/build_features.py)
методом Triple Barrier (processing/labeling.py) и сохраняет результат в
data/features/<root>_bars_labeled.csv.

Запуск:
    .venv/bin/python -m scripts.build_labels --root CNYRUBF
"""
import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processing.features import FEATURE_COLUMNS
from processing.labeling import LABEL_COLUMNS, compute_uniqueness_weights, overlap_stats, scan_barriers

NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume", "dollar_volume", "n_ticks",
    "days_to_expiration", "life_fraction_remaining", "contract_rank", "volume_share",
    "basis_pct_rub_only", "basis_pct_full", "log_return_1", "volatility_20", "momentum_10",
]


def read_features_csv(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for col in NUMERIC_COLUMNS:
            v = r.get(col)
            r[col] = float(v) if v not in (None, "") else None
    return rows


def group_by_ticker(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["ticker"]].append(r)  # исходный csv отсортирован по time -> порядок внутри тикера уже хронологический
    return g


def main(argv=None):
    parser = argparse.ArgumentParser(description="Triple Barrier поверх уже построенных баров/признаков.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--horizon-bars", type=int, default=20)
    parser.add_argument("--tp-vol-mult", type=float, default=2.0)
    parser.add_argument("--sl-vol-mult", type=float, default=1.0)
    args = parser.parse_args(argv)

    in_path = Path(__file__).resolve().parent.parent / "data" / "features" / f"{args.root}_bars_features.csv"
    if not in_path.exists():
        raise SystemExit(f"Нет файла {in_path} — сначала запусти scripts.build_features --root {args.root}")

    rows = read_features_csv(in_path)
    by_ticker = group_by_ticker(rows)
    print(f"{args.root}: {len(rows)} баров, {len(by_ticker)} контрактов. "
          f"Горизонт {args.horizon_bars} баров, TP={args.tp_vol_mult}xvol, SL={args.sl_vol_mult}xvol.")

    labels_by_ticker = {}
    for ticker, bars in by_ticker.items():
        labels = scan_barriers(
            bars, horizon_bars=args.horizon_bars,
            tp_vol_mult=args.tp_vol_mult, sl_vol_mult=args.sl_vol_mult,
        )
        compute_uniqueness_weights(labels, n_bars=len(bars))
        labels_by_ticker[ticker] = labels

    all_labels = [row for rows_ in labels_by_ticker.values() for row in rows_]
    all_labels.sort(key=lambda r: r["time"])
    print(f"\nРазмечено {len(all_labels)} из {len(rows)} баров "
          f"({len(all_labels) / max(len(rows), 1) * 100:.1f}% — остальные без volatility_20 "
          f"или без полного горизонта до конца контракта).")

    reason_counts = Counter(r["exit_reason"] for r in all_labels)
    print("\nРаспределение исхода:")
    for reason in ("tp", "sl", "timeout"):
        n = reason_counts.get(reason, 0)
        print(f"  {reason:8} {n:7} ({n / max(len(all_labels), 1) * 100:5.1f}%)")

    target_rate = sum(r["target"] for r in all_labels) / max(len(all_labels), 1)
    print(f"\nДоля target=1 (TP сработал первым): {target_rate * 100:.1f}%")

    print("\nbars_held / ret_gross по исходу:")
    for reason in ("tp", "sl", "timeout"):
        subset = [r for r in all_labels if r["exit_reason"] == reason]
        if not subset:
            continue
        held = [r["bars_held"] for r in subset]
        rets = [r["ret_gross"] for r in subset]
        print(f"  {reason:8} n={len(subset):6}  bars_held mean={statistics.mean(held):5.1f}  "
              f"ret_gross mean={statistics.mean(rets):+.4f} std={statistics.pstdev(rets):.4f}")

    stats = overlap_stats(labels_by_ticker)
    print(f"\nНахлёст окон меток (диагностика):")
    print(f"  mean overlap  = {stats['mean_overlap']:.1f} соседних меток на одну")
    print(f"  median overlap = {stats['median_overlap']:.1f}")

    weights = [r["sample_weight"] for r in all_labels]
    print(f"\nВеса по уникальности (compute_uniqueness_weights, компенсация нахлёста выше):")
    print(f"  mean={statistics.mean(weights):.3f}  median={statistics.median(weights):.3f}  "
          f"min={min(weights):.3f}  max={max(weights):.3f}")
    print(f"  (справочно: mean overlap {stats['mean_overlap']:.1f} -> примерно "
          f"1/{stats['mean_overlap']+1:.1f} ≈ {1/(stats['mean_overlap']+1):.3f} ожидаемо близко к mean-весу)")

    out_columns = FEATURE_COLUMNS + LABEL_COLUMNS
    out_path = in_path.parent / f"{args.root}_bars_labeled.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_columns)
        writer.writeheader()
        for row in all_labels:
            writer.writerow({k: row.get(k) for k in out_columns})
    print(f"\nСохранено: {out_path} ({len(all_labels)} строк).")


if __name__ == "__main__":
    main()
