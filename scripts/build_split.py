"""
Этап 1 — train/test-разбиение с purge/embargo (processing/splitting.py)
поверх уже размеченного датасета (data/features/<root>_bars_labeled.csv,
см. scripts/build_labels.py). Сохраняет data/features/<root>_train.csv и
<root>_test.csv.

Запуск:
    .venv/bin/python -m scripts.build_split --root CNYRUBF
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processing.features import FEATURE_COLUMNS
from processing.labeling import LABEL_COLUMNS
from processing.splitting import compute_split_time, get_root_time_range, purge_embargo_split

OUT_COLUMNS = FEATURE_COLUMNS + LABEL_COLUMNS


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train/test-разбиение с purge/embargo.")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="ДОЛЖНО совпадать с --test-fraction в scripts/build_features.py — та же граница "
             "использовалась при train-only калибровке порога dollar bars.",
    )
    parser.add_argument("--embargo-fraction", type=float, default=0.01)
    parser.add_argument("--tag", default="", help="суффикс файлов варианта разметки (см. build_labels --tag)")
    parser.add_argument(
        "--split-root", default=None,
        help="по какой root-серии брать диапазон свечей для границы train/test. "
             "Нужно для производных рядов, которых нет в БД как серии: у "
             "непрерывного ряда CNYRUBF_CONT (ETF Trick) свечей в futures_candles "
             "нет, а граница должна быть ТА ЖЕ, что у CNYRUBF — иначе сравнение "
             "непрерывного ряда с пулом пойдёт по разным периодам.",
    )
    args = parser.parse_args(argv)

    in_path = Path(__file__).resolve().parent.parent / "data" / "features" / f"{args.root}_bars_labeled{args.tag}.csv"
    if not in_path.exists():
        raise SystemExit(f"Нет файла {in_path} — сначала запусти scripts.build_labels --root {args.root}")

    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Та же граница, что и в scripts/build_features.py (не пересчитывается из
    # строк файла) — см. docstring processing/splitting.py.
    t_min, t_max = get_root_time_range(args.split_root or args.root)
    split_time = compute_split_time(t_min, t_max, args.test_fraction)

    train, test, report = purge_embargo_split(
        rows, split_time, t_min, t_max, embargo_fraction=args.embargo_fraction
    )

    print(f"{args.root}: {report['n_total']} размеченных строк, "
          f"{report['t_min'].date()} … {report['t_max'].date()}.")
    print(f"Граница test:   {report['split_time']}")
    print(f"Граница embargo: {report['embargo_start']} (буфер {args.embargo_fraction*100:.1f}% диапазона)")
    print()
    print(f"train: {report['n_train']:7} строк")
    print(f"  из них убрано purge   (t1 залезает в test): {report['n_purged']:6}")
    print(f"  из них убрано embargo (буфер перед test):    {report['n_embargoed']:6}")
    print(f"test:  {report['n_test']:7} строк")

    # Инвариант: ни один train-пример не должен "видеть" момент начала test
    from datetime import datetime
    max_t1_train = max((datetime.fromisoformat(r["t1"]) for r in train), default=None)
    ok = max_t1_train is None or max_t1_train < report["split_time"]
    print(f"\nПроверка: max(t1) по train = {max_t1_train}  (должно быть < граница test) — {'OK' if ok else 'НАРУШЕНО'}")
    assert ok, "purge не сработал: train содержит окно предсказания, залезающее в test"

    def target_rate(rows_):
        vals = [int(r["target"]) for r in rows_ if r.get("target") not in (None, "")]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\nДоля target=1: train {target_rate(train)*100:.1f}%   test {target_rate(test)*100:.1f}%")

    train_tickers = Counter(r["ticker"] for r in train)
    test_tickers = Counter(r["ticker"] for r in test)
    only_train = set(train_tickers) - set(test_tickers)
    only_test = set(test_tickers) - set(train_tickers)
    print(f"\nКонтрактов только в train: {len(only_train)}   только в test: {len(only_test)}"
          f"   (ожидаемо — контракты разных лет живут в разных периодах)")

    out_dir = in_path.parent
    for name, subset in (("train", train), ("test", test)):
        out_path = out_dir / f"{args.root}_{name}{args.tag}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
            writer.writeheader()
            for row in subset:
                writer.writerow({k: row.get(k) for k in OUT_COLUMNS})
        print(f"Сохранено: {out_path} ({len(subset)} строк)")


if __name__ == "__main__":
    main()
