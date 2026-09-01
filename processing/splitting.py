"""
Этап 1 — train/test-разбиение с purge и embargo (López de Prado, "Advances
in Financial Machine Learning" — plan.md, раздел 4).

Зачем это нужно именно здесь: метки (processing/labeling.py) размечены на
КАЖДОМ баре, а не на независимых событиях — окна предсказания (`time`..`t1`)
соседних меток сильно перекрываются (раздел 11 plan.md, mean overlap 7.8).
Простой разрез по дате пропустит в train примеры, чьё окно предсказания
залезает в test — модель увидит часть test-будущего уже на train. Отсюда:

* PURGE — из train убираются все строки, у которых t1 >= момент начала test
  (окно предсказания залезает в test). Точный механизм, по фактическим
  границам меток.
* EMBARGO — дополнительно убирается буфер train сразу перед началом test
  (доля от общего временного диапазона), сверх того, что убрал purge —
  подстраховка от более тонкой автокорреляции, которую точная граница
  окна метки не ловит.

Разрез — ПО ВРЕМЕНИ (не по числу строк): строки неравномерно плотные во
времени (в поздние годы одновременно живёт больше контрактов, барах гуще) —
разрез по числу строк сместил бы test в сторону последних лет непропорционально.

Один разрез (train раньше test), не k-fold — для первого прохода этого
достаточно.

Граница `split_time` считается из диапазона СЫРЫХ СВЕЧЕЙ серии
(`get_root_time_range` + `compute_split_time`), а не из размеченных строк —
это специально: та же самая граница используется в scripts/build_features.py
для train-only калибровки порога dollar bars (раздел 10.1 plan.md), и она
должна СОВПАДАТЬ с границей, которая потом реально разрезает train/test —
иначе калибровка порога и сам разрез будут использовать два разных
"train", и защита от утечки станет фиктивной. Оба скрипта обязаны получать
одинаковый --test-fraction.
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db.database import engine


def get_root_time_range(root_symbol: str) -> Tuple[datetime, datetime]:
    """Диапазон времени сырых свечей root-серии — источник границы split_time."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT MIN(fc.time) AS t_min, MAX(fc.time) AS t_max
                FROM futures_candles fc
                JOIN futures_contracts c ON c.figi = fc.figi
                WHERE c.root_symbol = :root
                """
            ),
            {"root": root_symbol},
        ).fetchone()
    return row.t_min, row.t_max


def compute_split_time(t_min: datetime, t_max: datetime, test_fraction: float = 0.2) -> datetime:
    return t_min + (t_max - t_min) * (1 - test_fraction)


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def purge_embargo_split(
    rows: List[dict],
    split_time: datetime,
    t_min: datetime,
    t_max: datetime,
    embargo_fraction: float = 0.01,
) -> Tuple[List[dict], List[dict], dict]:
    """
    rows — размеченные строки (processing/labeling.py: поля 'time', 't1'),
    в любом порядке.

    split_time/t_min/t_max передаются ЯВНО (см. get_root_time_range +
    compute_split_time) — та же граница, что использовалась при
    калибровке порога dollar bars, не пересчитывается заново из rows.

    Возвращает (train, test, report). report — числа для диагностики: сколько
    строк убрано purge, сколько embargo, границы времени, потому что "просто
    сколько получилось" не заменяет проверку, что разрез сделан честно.
    """
    span = t_max - t_min
    embargo_gap = span * embargo_fraction
    embargo_start = split_time - embargo_gap

    train, test = [], []
    n_purged, n_embargoed = 0, 0
    for r in rows:
        t = _parse_dt(r["time"])
        if t >= split_time:
            test.append(r)
            continue
        # кандидат в train — сначала purge (по t1), потом embargo (по времени бара)
        t1 = _parse_dt(r["t1"]) if r.get("t1") else t
        if t1 >= split_time:
            n_purged += 1
            continue
        if t >= embargo_start:
            n_embargoed += 1
            continue
        train.append(r)

    report = {
        "t_min": t_min, "t_max": t_max,
        "split_time": split_time, "embargo_start": embargo_start,
        "n_total": len(rows), "n_train": len(train), "n_test": len(test),
        "n_purged": n_purged, "n_embargoed": n_embargoed,
    }
    return train, test, report
