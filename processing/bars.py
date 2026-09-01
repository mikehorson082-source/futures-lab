"""
Этап 1 (CNYRUBF, но код общий на любую root-серию) — бары по накопленному
обороту (dollar bars): бар закрывается не по времени, а когда накопленный
оборот (close*volume) превышает порог — так бар несёт примерно одинаковое
количество реальной торговой активности вне зависимости от дня недели.

Бары строятся НА КАЖДЫЙ КОНТРАКТ ОТДЕЛЬНО — накопитель оборота не
переносится через ролл на следующий контракт, иначе это была бы та самая
склейка, от которой в разделе 3 plan.md явно отказались.

Порог оборота — калибруется как ФУНКЦИЯ от days_to_expiration (не число
на контракт), пулом по всем контрактам сразу. Так исторически сложилось
не сразу: первая версия считала один порог на контракт (средний дневной
оборот всей его известной жизни), но у контрактов, чья ликвидная фаза
(последние дни перед экспирацией — раздел 2 plan.md, там же концентрация
объёма) приходится на test-период, "своя" train-история — это только
тонкое начало жизни. Порог, откалиброванный по тонкому началу и
применённый ко всей жизни контракта, давал у таких контрактов вместо
целевых ~30 баров/день — 250-275: бар не переходил на границе train/test
как надо, а просто резался в разы чаще на ликвидном хвосте. Найдено и
явно обсуждено с пользователем (см. plan.md, раздел 12).

Решение — считать не число на контракт, а КРИВУЮ порога по бакетам
days_to_expiration (те же бакеты, что в scripts/diagnose_roll_splice.py,
раздел 2): пороговое значение для бакета — средний дневной оборот ВСЕХ
контрактов на этом расстоянии до экспирации, посчитанный ТОЛЬКО по
train-периоду (`before`). Молодой контракт, чья собственная ликвидная
фаза ещё не наступила, получает порог из чужого — уже отторговавшего —
опыта на том же расстоянии до экспирации, а не растянутую на всю жизнь
свою тонкую историю. Реализовано в `calibrate_threshold_curve` +
`thresholds_for_contract`.

Для диагностики механики баров (scripts/build_bars.py, где важна форма
распределения длительности бара, а не отсутствие утечки) оставлена
простая `calibrate_threshold` — единый порог по всей истории контракта.

Граница бара — торговый сегмент по db/calendar.py (разрыв между соседними
минутными свечами > GAP_THRESHOLD_MINUTES), не календарная дата: пятница /
ДСВД-суббота / ДСВД-воскресенье / понедельник уже расходятся на разные
сегменты сами по себе (см. plan.md, раздел 7.5) — этой логикой пользуемся
напрямую, отдельно её не переизобретаем.
"""
import bisect
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


DEFAULT_DTE_BUCKETS = [7, 30, 90, 180]  # те же границы, что в scripts/diagnose_roll_splice.py


def _bucket_index(days_to_expiration: int, buckets: List[int] = DEFAULT_DTE_BUCKETS) -> int:
    """0 = '<=7д', 1 = '7-30д', ..., len(buckets) = '>180д'."""
    for i, b in enumerate(buckets):
        if days_to_expiration <= b:
            return i
    return len(buckets)


def calibrate_threshold_curve(
    candles_by_ticker: Dict[str, List[Candle]],
    seg_ids_by_ticker: Dict[str, List[int]],
    expiration_by_ticker: Dict[str, date],
    target_bars_per_day: int = 30,
    before: Optional[datetime] = None,
    buckets: List[int] = DEFAULT_DTE_BUCKETS,
) -> List[Optional[float]]:
    """
    Порог dollar bars как ФУНКЦИЯ от бакета days_to_expiration (не число
    на контракт) — пул ВСЕХ контрактов серии сразу, только train-период
    (дни строго ДО `before`). См. докстринг модуля — так молодой контракт
    без собственной ликвидной истории получает порог из чужого опыта на
    том же расстоянии до экспирации.

    Возвращает список длины len(buckets)+1 — threshold[bucket_index].
    Пустой бакет (в train не нашлось ни одного дня на этом расстоянии до
    экспирации ни у одного контракта) заполняется ближайшим непустым
    соседним бакетом — лучше грубое приближение, чем None и потерянные бары.
    """
    bucket_dv: Dict[int, float] = defaultdict(float)
    bucket_days: Dict[int, int] = defaultdict(int)

    for ticker, candles in candles_by_ticker.items():
        seg_ids = seg_ids_by_ticker[ticker]
        exp = expiration_by_ticker[ticker]
        daily_dv: Dict[int, float] = defaultdict(float)
        daily_date = {}
        for (t, close, vol), seg in zip(candles, seg_ids):
            if before is not None and t >= before:
                continue
            daily_dv[seg] += close * vol
            daily_date[seg] = t.date()
        for seg, dv in daily_dv.items():
            dte = (exp - daily_date[seg]).days
            b = _bucket_index(dte, buckets)
            bucket_dv[b] += dv
            bucket_days[b] += 1

    n_buckets = len(buckets) + 1
    thresholds: List[Optional[float]] = [None] * n_buckets
    for b in range(n_buckets):
        if bucket_days.get(b, 0) > 0:
            thresholds[b] = max(bucket_dv[b] / bucket_days[b] / target_bars_per_day, 1.0)

    # Заполнить пустые бакеты ближайшим непустым соседом (по индексу бакета)
    for b in range(n_buckets):
        if thresholds[b] is not None:
            continue
        for d in range(1, n_buckets):
            left, right = b - d, b + d
            if 0 <= left < n_buckets and thresholds[left] is not None:
                thresholds[b] = thresholds[left]
                break
            if 0 <= right < n_buckets and thresholds[right] is not None:
                thresholds[b] = thresholds[right]
                break

    return thresholds


def thresholds_for_contract(
    candles: List[Candle],
    expiration_date: date,
    bucket_thresholds: List[Optional[float]],
    buckets: List[int] = DEFAULT_DTE_BUCKETS,
) -> List[float]:
    """Порог для КАЖДОЙ свечи контракта — по бакету days_to_expiration на дату свечи."""
    out = []
    for t, _, _ in candles:
        dte = (expiration_date - t.date()).days
        b = _bucket_index(dte, buckets)
        out.append(bucket_thresholds[b])
    return out


def build_dollar_bars(candles: List[Candle], seg_ids: List[int], threshold) -> List[dict]:
    """
    Нарезка: бар закрывается, когда накопленный оборот (без текущей свечи)
    достиг порога, ИЛИ на границе торгового сегмента (форсированное закрытие
    по дню). Свеча, которая пробивает порог, остаётся в текущем баре — новый
    бар стартует со следующей свечи.

    `threshold` — либо одно число (весь контракт, scripts/build_bars.py),
    либо список той же длины, что `candles` — порог для КАЖДОЙ свечи (см.
    thresholds_for_contract), меняющийся по мере приближения к экспирации.
    """
    per_candle = isinstance(threshold, (list, tuple))
    bars: List[dict] = []
    cur: Optional[dict] = None
    cum_dv = 0.0
    cur_seg = None

    for idx, ((t, close, vol), seg) in enumerate(zip(candles, seg_ids)):
        thr = threshold[idx] if per_candle else threshold
        if cur is None or seg != cur_seg or cum_dv >= thr:
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
