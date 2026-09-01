"""
Сегментация свечей на непрерывные отрезки торговли ("торговые дни" по
факту) — без завязки на конкретные времена сессий FORTS.

См. plan.md: константы сессии (утро/день/клиринг/вечер/ДСВД) менялись
минимум 5 раз за 2025-2026 год, поэтому граница торгового дня определяется
не по расписанию, а по разрыву между соседними минутными свечами.

Порог GAP_THRESHOLD_MINUTES подобран по фактическому распределению
разрывов (проверено на всех 5 сериях, 2022-2026): между "обычным" шумом
(клиринг, тонкая ликвидность — до ~215 минут в редких случаях, каждый из
которых разобран и объясняется реальным событием: 24.02.2022 — начало СВО,
25.03-18.04.2022 — первые недели после открытия биржи, 13.09.2023 —
общерыночная пауза одновременно во всех 5 сериях) и настоящими
межсессионными разрывами (от ~430 минут, будни-выходные и обычные ночи) —
пустой промежуток. Конкретное число порога внутри него не критично.
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db.database import engine

GAP_THRESHOLD_MINUTES = 180


class TradingSegment(NamedTuple):
    start: datetime
    end: datetime


def get_trading_day_segments(root_symbol: str, gap_threshold_minutes: int = GAP_THRESHOLD_MINUTES) -> List[TradingSegment]:
    """
    Непрерывные отрезки торговли для root-серии (объединяет свечи всех её
    контрактов по времени). Один отрезок может охватывать несколько
    календарных дат — например, пятница -> ДСВД суббота-воскресенье ->
    понедельник, если реальные разрывы между ними меньше порога. Это
    следствие данных, а не предположение о конвенции биржи, заложенное
    заранее в код.
    """
    query = text(
        """
        WITH ts AS (
            SELECT DISTINCT fc.time AS t
            FROM futures_candles fc
            JOIN futures_contracts ct ON ct.figi = fc.figi
            WHERE ct.root_symbol = :root
        ),
        gapped AS (
            SELECT
                t,
                EXTRACT(EPOCH FROM (t - LAG(t) OVER (ORDER BY t))) / 60.0 AS gap_min
            FROM ts
        ),
        segmented AS (
            SELECT
                t,
                SUM(CASE WHEN gap_min IS NULL OR gap_min > :threshold THEN 1 ELSE 0 END)
                    OVER (ORDER BY t) AS segment_id
            FROM gapped
        )
        SELECT segment_id, MIN(t) AS seg_start, MAX(t) AS seg_end
        FROM segmented
        GROUP BY segment_id
        ORDER BY segment_id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"root": root_symbol, "threshold": gap_threshold_minutes}).fetchall()
    return [TradingSegment(r.seg_start, r.seg_end) for r in rows]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Диагностика: сегменты торговли по root-серии.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--threshold", type=int, default=GAP_THRESHOLD_MINUTES)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    segments = get_trading_day_segments(args.root, args.threshold)
    print(f"Root {args.root}: {len(segments)} сегментов при пороге {args.threshold} мин.")
    for seg in segments[: args.limit]:
        duration_h = (seg.end - seg.start).total_seconds() / 3600
        print(f"  {seg.start} -> {seg.end}  (длительность {duration_h:.1f} ч)")
