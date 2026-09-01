"""
Этап 1 (CNYRUBF, но код общий на любую root-серию) — бары по накопленному
обороту (dollar bars). По образцу ../agent/processing/bar_sampler.py, но
адаптировано под архитектуру futures-lab: пул раздельных контрактов
(plan.md, раздел 3), а не склеенный непрерывный ряд.

Ключевое отличие от agent: бары строятся НА КАЖДЫЙ КОНТРАКТ ОТДЕЛЬНО —
накопитель оборота не переносится через ролл на следующий контракт, иначе
это была бы та самая склейка, от которой в разделе 3 явно отказались.

Порог оборота — калибруется на контракт: средний дневной оборот
(close*volume) за всю известную жизнь контракта, делённая на
target_bars_per_day.

⚠️ Провизорное решение (первый проход): порог считается по ВСЕЙ истории
контракта, не только по train-периоду, как в ../agent
(`calibrate_thresholds` там явно ограничен train, "иначе порог знал бы о
будущих объёмах"). Здесь это пока не сделано — train/test-разбиение для
futures-lab ещё не определено. Если дойдём до обучения модели, порог
нужно будет пересчитать по train-части, иначе он неявно использует
информацию о будущих объёмах того же контракта. Зафиксировано явно, не
спрятано (см. CLAUDE.md, п.4).

Граница бара — торговый сегмент по db/calendar.py (разрыв между соседними
минутными свечами > GAP_THRESHOLD_MINUTES), не календарная дата: пятница /
ДСВД-суббота / ДСВД-воскресенье / понедельник уже расходятся на разные
сегменты сами по себе (см. plan.md, раздел 7.5) — этой логикой пользуемся
напрямую, отдельно её не переизобретаем.
"""
import bisect
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db.calendar import TradingSegment
from db.database import engine

Candle = Tuple[datetime, float, int]  # (time, close, volume)


def get_contract_candles(figi: str) -> List[Candle]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT time, close, volume FROM futures_candles WHERE figi = :figi ORDER BY time"),
            {"figi": figi},
        ).fetchall()
    return [(r.time, float(r.close), int(r.volume)) for r in rows]


def assign_segments(candle_times: List[datetime], segments: List[TradingSegment]) -> List[int]:
    """Индекс торгового сегмента для каждой свечи (бинарный поиск по началу сегмента)."""
    starts = [s.start for s in segments]
    return [bisect.bisect_right(starts, t) - 1 for t in candle_times]


def calibrate_threshold(
    candles: List[Candle], seg_ids: List[int], target_bars_per_day: int = 30
) -> Optional[float]:
    """Средний дневной оборот контракта (по торговым сегментам) / target_bars_per_day.

    См. докстринг модуля — калибровка по всей истории контракта, не только train.
    """
    daily_dv = defaultdict(float)
    for (_, close, vol), seg in zip(candles, seg_ids):
        daily_dv[seg] += close * vol
    if not daily_dv:
        return None
    avg_daily_dv = sum(daily_dv.values()) / len(daily_dv)
    return max(avg_daily_dv / target_bars_per_day, 1.0)


def build_dollar_bars(candles: List[Candle], seg_ids: List[int], threshold: float) -> List[dict]:
    """
    Нарезка: бар закрывается, когда накопленный оборот (без текущей свечи)
    достиг порога, ИЛИ на границе торгового сегмента (форсированное закрытие
    по дню). Свеча, которая пробивает порог, остаётся в текущем баре — новый
    бар стартует со следующей свечи (тот же принцип, что в agent).
    """
    bars: List[dict] = []
    cur: Optional[dict] = None
    cum_dv = 0.0
    cur_seg = None

    for (t, close, vol), seg in zip(candles, seg_ids):
        if cur is None or seg != cur_seg or cum_dv >= threshold:
            if cur is not None:
                bars.append(cur)
            cur = {
                "open_time": t, "time": t,
                "open": close, "high": close, "low": close, "close": close,
                "volume": 0, "dollar_volume": 0.0, "n_ticks": 0,
            }
            cum_dv = 0.0
            cur_seg = seg

        dv = close * vol
        cur["high"] = max(cur["high"], close)
        cur["low"] = min(cur["low"], close)
        cur["close"] = close
        cur["volume"] += vol
        cur["dollar_volume"] += dv
        cur["n_ticks"] += 1
        cur["time"] = t
        cum_dv += dv

    if cur is not None:
        bars.append(cur)
    return bars
