"""
Этап 1 — один запуск на весь пайплайн подготовки датасета для root-серии:

    свечи -> dollar bars -> признаки       (scripts/build_features.py)
          -> разметка Triple Barrier       (scripts/build_labels.py)
          -> train/test с purge/embargo    (scripts/build_split.py)
          -> стационарные признаки v2      (scripts/build_derived_features.py)

Четыре шага остаются самостоятельными скриптами (каждый можно перезапустить
отдельно, например, только пересчитать разметку с другим горизонтом) —
этот файл их просто вызывает по очереди с согласованными аргументами, не
дублирует логику. Важно согласовано, а не по отдельности: --test-fraction
должен быть ОДИН и тот же на шаге признаков (там калибруется train-only
порог dollar bars) и на шаге разреза (там он же режет train/test) — при
раздельном запуске это нужно помнить руками, здесь передаётся один раз.
--tag тоже один на весь пайплайн (см. раздел 24 plan.md) — так варианты
разметки (например, широкие барьеры) не затирают друг друга на диске.

Результат — шесть файлов в data/features/ (не в git):
    <root>_bars_features.csv     — бары + признаки (шаг 1)
    <root>_bars_labeled<tag>.csv — + разметка Triple Barrier (шаг 2)
    <root>_train<tag>.csv        — train после purge/embargo (шаг 3)
    <root>_test<tag>.csv         — test после purge/embargo (шаг 3)
    <root>_train<tag>_v2.csv     — train + стационарные признаки (шаг 4)
    <root>_test<tag>_v2.csv      — test + стационарные признаки (шаг 4)

Запуск (по умолчанию — узкие барьеры раздела 11, для честного baseline):
    .venv/bin/python -m scripts.build_dataset --root CNYRUBF

Запуск финальной конфигурации Этапа 1 (широкие барьеры, раздел 24):
    .venv/bin/python -m scripts.build_dataset --root CNYRUBF --tag _w4 \\
        --horizon-bars 80 --tp-vol-mult 4.0 --sl-vol-mult 2.0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import build_derived_features, build_features, build_labels, build_split


def main(argv=None):
    parser = argparse.ArgumentParser(description="Полный пайплайн: бары -> признаки -> разметка -> train/test -> v2.")
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
    parser.add_argument(
        "--tag", default="",
        help="Суффикс варианта разметки (например --tag _w4) — передаётся в шаги 2-4, "
             "чтобы не затирать файлы другого варианта на диске.",
    )
    args = parser.parse_args(argv)

    print(f"=== Шаг 1/4: бары + признаки ({args.root}) ===")
    build_features.main([
        "--root", args.root,
        "--target-bars-per-day", str(args.target_bars_per_day),
        "--min-candles", str(args.min_candles),
        "--test-fraction", str(args.test_fraction),
    ])

    print(f"\n=== Шаг 2/4: разметка Triple Barrier ({args.root}) ===")
    build_labels.main([
        "--root", args.root,
        "--horizon-bars", str(args.horizon_bars),
        "--tp-vol-mult", str(args.tp_vol_mult),
        "--sl-vol-mult", str(args.sl_vol_mult),
        "--tag", args.tag,
    ])

    print(f"\n=== Шаг 3/4: train/test с purge/embargo ({args.root}) ===")
    build_split.main([
        "--root", args.root,
        "--test-fraction", str(args.test_fraction),
        "--embargo-fraction", str(args.embargo_fraction),
        "--tag", args.tag,
    ])

    print(f"\n=== Шаг 4/4: стационарные признаки v2 ({args.root}) ===")
    build_derived_features.main([
        "--root", args.root,
        "--tag", args.tag,
    ])

    print(f"\n=== Готово: data/features/{args.root}_bars_features.csv, "
          f"{args.root}_bars_labeled{args.tag}.csv, "
          f"{args.root}_train{args.tag}.csv, {args.root}_test{args.tag}.csv, "
          f"{args.root}_train{args.tag}_v2.csv, {args.root}_test{args.tag}_v2.csv ===")


if __name__ == "__main__":
    main()
