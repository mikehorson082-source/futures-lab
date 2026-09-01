"""Добавляет стационарные производные признаки (processing.features.
DERIVED_FEATURES) к уже готовым train/test Этапа 1.

Считаются на ПОЛНОЙ размеченной таблице (не отдельно на train и test):
окна причинные (только прошлые бары того же контракта), поэтому заглядывания
в будущее нет, зато у test нет "прогревочной дыры" в начале — ровно так же
считала бы живая система.

Разбиение train/test берётся из уже посчитанных файлов (purge/embargo,
раздел 12) — новые колонки приклеиваются к тем же строкам по (ticker, time).

Запуск:
    .venv/bin/python -m scripts.build_derived_features --root CNYRUBF
"""
import argparse

import pandas as pd

from processing.features import DERIVED_FEATURES, add_derived_features


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="CNYRUBF")
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)

    full = pd.read_csv(f"data/features/{a.root}_bars_labeled{a.tag}.csv", parse_dates=["time"])
    full = add_derived_features(full)
    print(f"полная таблица: {len(full)} строк")
    for c in DERIVED_FEATURES:
        print(f"  {c:14s} покрытие {full[c].notna().mean():6.1%}")

    keys = ["ticker", "time"]
    derived = full[keys + DERIVED_FEATURES]
    for part in ("train", "test"):
        df = pd.read_csv(f"data/features/{a.root}_{part}{a.tag}.csv", parse_dates=["time"])
        out = df.merge(derived, on=keys, how="left", validate="one_to_one")
        assert len(out) == len(df)
        path = f"data/features/{a.root}_{part}{a.tag}_v2.csv"
        out.to_csv(path, index=False)
        print(f"{part}: {len(out)} строк -> {path}")


if __name__ == "__main__":
    main()
