"""
Этап 1 — один запуск на весь пайплайн подготовки датасета для root-серии:

    свечи -> dollar bars -> признаки       (scripts/build_features.py)
          -> разметка Triple Barrier       (scripts/build_labels.py)
          -> train/test с purge/embargo    (scripts/build_split.py)

Три шага остаются самостоятельными скриптами (каждый можно перезапустить
отдельно, например, только пересчитать разметку с другим горизонтом) —
этот файл их просто вызывает по очереди с согласованными аргументами, не
дублирует логику. Важно согласовано, а не по отдельности: --test-fraction
должен быть ОДИН и тот же на шаге признаков (там калибруется train-only
порог dollar bars) и на шаге разреза (там он же режет train/test) — при
раздельном запуске это нужно помнить руками, здесь передаётся один раз.

Результат — четыре файла в data/features/ (не в git):
    <root>_bars_features.csv   — бары + признаки (шаг 1)
    <root>_bars_labeled.csv    — + разметка Triple Barrier (шаг 2)
    <root>_train.csv           — train после purge/embargo (шаг 3)
    <root>_test.csv            — test после purge/embargo (шаг 3)

Запуск:
    .venv/bin/python -m scripts.build_dataset --root CNYRUBF
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import build_features, build_labels, build_split


def main(argv=None):
    parser = argparse.ArgumentParser(description="Полный пайплайн: бары -> признаки -> разметка -> train/test.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--target-bars-per-day", type=int, default=30)
    parser.add_argument("--min-candles", type=int, default=50)
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Общая граница train/test — используется и для калибровки порога dollar bars "
             "(шаг 1), и для самого разреза (шаг 3). Одно значение на весь пайплайн.",
    )
    parser.add_argument("--embargo-fraction", type=float, default=0.01)
    parser.add_argument("--horizon-bars", type=int, default=20)
    parser.add_argument("--tp-vol-mult", type=float, default=2.0)
    parser.add_argument("--sl-vol-mult", type=float, default=1.0)
    args = parser.parse_args(argv)

    print(f"=== Шаг 1/3: бары + признаки ({args.root}) ===")
    build_features.main([
        "--root", args.root,
        "--target-bars-per-day", str(args.target_bars_per_day),
        "--min-candles", str(args.min_candles),
        "--test-fraction", str(args.test_fraction),
    ])

    print(f"\n=== Шаг 2/3: разметка Triple Barrier ({args.root}) ===")
    build_labels.main([
        "--root", args.root,
        "--horizon-bars", str(args.horizon_bars),
        "--tp-vol-mult", str(args.tp_vol_mult),
        "--sl-vol-mult", str(args.sl_vol_mult),
    ])

    print(f"\n=== Шаг 3/3: train/test с purge/embargo ({args.root}) ===")
    build_split.main([
        "--root", args.root,
        "--test-fraction", str(args.test_fraction),
        "--embargo-fraction", str(args.embargo_fraction),
    ])

    print(f"\n=== Готово: data/features/{args.root}_bars_features.csv, "
          f"{args.root}_bars_labeled.csv, {args.root}_train.csv, {args.root}_test.csv ===")


if __name__ == "__main__":
    main()
