"""
Эксперимент «все фьючерсы в одном датасете» (по просьбе пользователя,
2026-09-02).

Идея: до сих пор каждая root-серия жила отдельно — свои бары, своя
разметка, своя модель. Здесь мы проверяем гипотезу переноса: есть ли в
барах общий механизм, который модель может выучить на четырёх рынках
сразу (валюта, акция, индекс, нефть) и который работает лучше, чем
модель, обученная только на своём инструменте.

Что делает скрипт:
    1. Берёт УЖЕ посчитанные размеченные таблицы <root>_bars_labeled<tag>.csv
       (bars -> features -> triple barrier), т.е. ничего не пересчитывает.
    2. Добавляет производные (стационарные) признаки — на полной таблице
       КАЖДОГО root отдельно (окна причинные и внутри контракта).
    3. Склеивает в один пул, добавляя колонку root_symbol как явный признак
       происхождения строки (использовать её в модели или нет — решает
       обучающий скрипт).
    4. Режет train/test ОДНОЙ общей границей по времени для всех
       инструментов. Это принципиально: если у каждого root своя граница,
       train одного инструмента залезает в test другого, и пул начинает
       видеть будущий режим рынка через соседний рынок. Общая граница =
       compute_split_time по ОБЪЕДИНЁННОМУ диапазону сырых свечей.
    5. purge/embargo — тем же кодом processing/splitting.py, что и в
       одиночном пайплайне.

Запуск:
    .venv/bin/python -m scripts.build_pooled_dataset \
        --roots CNYRUBF SBERF IMOEXF BR --tag _w4
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processing.features import DERIVED_FEATURES, FEATURE_COLUMNS, add_derived_features
from processing.labeling import LABEL_COLUMNS
from processing.splitting import compute_split_time, get_root_time_range, purge_embargo_split

OUT_COLUMNS = FEATURE_COLUMNS + LABEL_COLUMNS + DERIVED_FEATURES
DATA = Path(__file__).resolve().parent.parent / "data" / "features"


def main(argv=None):
    p = argparse.ArgumentParser(description="Склейка нескольких root-серий в один датасет.")
    p.add_argument("--roots", nargs="+", default=["CNYRUBF", "SBERF", "IMOEXF", "BR"])
    p.add_argument("--tag", default="_w4")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--embargo-fraction", type=float, default=0.01)
    p.add_argument("--out", default="POOL")
    a = p.parse_args(argv)

    # --- общая граница по объединённому диапазону сырых свечей
    ranges = {r: get_root_time_range(r) for r in a.roots}
    t_min = min(v[0] for v in ranges.values())
    t_max = max(v[1] for v in ranges.values())
    split_time = compute_split_time(t_min, t_max, a.test_fraction)
    print(f"объединённый диапазон: {t_min.date()} … {t_max.date()}")
    print(f"ОБЩАЯ граница train/test: {split_time}\n")

    parts = []
    for root in a.roots:
        path = DATA / f"{root}_bars_labeled{a.tag}.csv"
        if not path.exists():
            raise SystemExit(f"нет {path} — сначала build_features + build_labels для {root}")
        df = pd.read_csv(path, parse_dates=["time"])
        df = add_derived_features(df)
        df["root_symbol"] = root
        cov = " ".join(f"{c}={df[c].notna().mean():.0%}" for c in DERIVED_FEATURES)
        print(f"{root:8s} {len(df):7d} строк  {df['time'].min().date()}…{df['time'].max().date()}  "
              f"target=1 {df['target'].mean():.1%}")
        print(f"         покрытие производных: {cov}")
        parts.append(df)

    pool = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
    for c in OUT_COLUMNS:
        if c not in pool.columns:
            pool[c] = None
    pool = pool[OUT_COLUMNS]

    # purge/embargo — тем же кодом, что и в одиночном пайплайне
    rows = pool.assign(
        time=pool["time"].astype(str), t1=pool["t1"].astype(str)
    ).to_dict("records")
    train, test, rep = purge_embargo_split(rows, split_time, t_min, t_max, a.embargo_fraction)
    print(f"\npool: {rep['n_total']} строк -> train {rep['n_train']}, test {rep['n_test']} "
          f"(purge {rep['n_purged']}, embargo {rep['n_embargoed']}; "
          f"embargo с {rep['embargo_start']})")

    for name, subset in (("train", train), ("test", test)):
        df = pd.DataFrame(subset, columns=OUT_COLUMNS)
        by_root = df.groupby("root_symbol").agg(rows=("target", "size"), target1=("target", "mean"))
        print(f"\n{name}:")
        for r, row in by_root.iterrows():
            print(f"  {r:8s} {int(row['rows']):7d} строк, target=1 {row['target1']:.1%}")
        out = DATA / f"{a.out}_{name}{a.tag}.csv"
        df.to_csv(out, index=False)
        print(f"  -> {out}")

    # инвариант: ни одна train-строка не заглядывает за границу test
    mx = pd.to_datetime(pd.DataFrame(train)["t1"]).max()
    assert mx < split_time, f"purge не сработал: max(t1) train = {mx}"
    print(f"\nПроверка: max(t1) по train = {mx} < {split_time} — OK")


if __name__ == "__main__":
    main()
