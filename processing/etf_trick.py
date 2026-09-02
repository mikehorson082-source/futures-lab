"""
ETF Trick (López de Prado, "Advances in Financial Machine Learning", §2.4.1)
— склейка серии фьючерсных контрактов в ОДИН непрерывный ряд.

Зачем. Контракт живёт несколько месяцев и умирает; чтобы иметь длинную
историю "рынка", контракты надо соединить. Наивное соединение встык врёт:
на стыке цена прыгает (у CNYRUBF в среднем 3.15%, максимум 6.93% — раздел
1 журнала), и этот прыжок выглядит как доходность, которую никто не
получал. Back-adjustment (вычесть разрыв из всей истории) чинит уровни, но
портит доходности и может увести цену в минус на длинной истории.

Идея ETF Trick: склеивать не ЦЕНУ, а стоимость самофинансируемой позиции —
"сколько стоил бы 1 рубль, вложенный в стратегию «держать ближний контракт
и роллировать»". Формально (§2.4.1), для каждого бара t:

    h_{t-1} = K_{t-1} / (o_t * φ)      — сколько контрактов держим
    δ_t     = p_t - p_{t-1}            — обычный бар (тот же контракт)
              p_t - o_t                — первый бар после ролла (новый контракт)
    K_t     = K_{t-1} + h_{t-1} * φ * δ_t

где K — NAV (наш непрерывный ряд), o — открытие бара, p — закрытие, φ —
стоимость пункта.

ВАЖНО и честно: для ОДНОГО инструмента без дивидендов φ сокращается
(K_t = K_{t-1}·(1 + δ_t/o_t)), и формула вырождается в "перемножение
доходностей ВНУТРИ контракта". Стоимость пункта, веса и корзина из
нескольких инструментов — та часть трюка, которая здесь не работает, потому
что инструмент один. Что реально даёт трюк в нашем случае:

  1. разрыв на ролле НЕ попадает в ряд ни как доходность, ни как уровень;
  2. ряд положителен по построению и интерпретируется как NAV (в отличие
     от back-adjusted ряда, который может уйти ниже нуля);
  3. издержки ролла можно вычесть явно и в правильном месте (roll_cost_bps),
     а не "забыть", как это происходит при склейке цен.

Модуль — чистые функции над списками свечей, без БД и без pandas: его
можно проверить на игрушечных данных (см. scripts/build_continuous_series.py
и его диагностику "разрыв, который трюк не пропустил в ряд").
"""
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple

# (time, close, volume) — тот же тип свечи, что в processing/bars.py
Candle = Tuple[datetime, float, int]


def build_roll_schedule_by_volume(
    daily_volume: Dict[str, Dict[date, float]],
    order: Sequence[str],
    expiration: Dict[str, date],
    last_candle_date: Dict[str, date],
) -> List[Tuple[date, str]]:
    """Расписание роллов по ПЕРЕТОКУ ЛИКВИДНОСТИ, причинно.

    Правило: если в день d оборот следующего контракта превысил оборот
    текущего — переходим со СЛЕДУЮЩЕГО торгового дня (решение принимается
    по уже закрытому дню d, не по дню перехода: иначе это заглядывание в
    будущее на один день).

    Дополнительно — принудительный переход, если у текущего контракта
    кончились свечи или наступила дата экспирации: держать истёкший
    контракт нельзя.

    Возвращает [(дата начала владения, тикер), ...] по возрастанию даты.
    """
    order = list(order)
    all_days = sorted({d for by_d in daily_volume.values() for d in by_d})
    if not all_days:
        return []

    idx = 0
    while idx < len(order) - 1 and not daily_volume.get(order[idx]):
        idx += 1
    schedule = [(all_days[0], order[idx])]
    pending: Optional[int] = None

    for d in all_days:
        cur = order[idx]
        if pending is not None:
            idx = pending
            pending = None
            schedule.append((d, order[idx]))
            cur = order[idx]
        if idx + 1 >= len(order):
            continue
        nxt = order[idx + 1]
        # принудительно: контракт истекает или кончились данные
        forced = expiration[cur] <= d or last_candle_date.get(cur, d) <= d
        v_cur = daily_volume.get(cur, {}).get(d, 0.0)
        v_nxt = daily_volume.get(nxt, {}).get(d, 0.0)
        if forced or (v_nxt > v_cur and v_nxt > 0):
            pending = idx + 1
    return schedule


def build_roll_schedule_by_days(
    order: Sequence[str],
    expiration: Dict[str, date],
    first_candle_date: Dict[str, date],
    days_before: int,
) -> List[Tuple[date, str]]:
    """Расписание роллов по календарю: перейти за `days_before` дней до экспирации."""
    schedule = []
    for i, t in enumerate(order):
        start = expiration[order[i - 1]] if i > 0 else first_candle_date.get(t)
        if i > 0:
            from datetime import timedelta
            start = expiration[order[i - 1]] - timedelta(days=days_before)
        if start is None:
            continue
        schedule.append((start, t))
    return sorted(schedule)


def apply_etf_trick(
    candles_by_ticker: Dict[str, List[Candle]],
    schedule: List[Tuple[date, str]],
    k0: float = 1.0,
    roll_cost_bps: float = 0.0,
) -> Tuple[List[dict], List[dict]]:
    """Собирает непрерывный ряд NAV по расписанию владения контрактами.

    Возвращает (series, rolls):
      series — список словарей на КАЖДУЮ минутную свечу удерживаемого
               контракта: time, ticker, raw_close (цена самого контракта),
               nav (K_t), volume, dollar_volume, is_roll;
      rolls  — диагностика по каждому переходу: дата, из какого в какой
               контракт, цена старого и нового в момент перехода и разрыв
               в процентах — тот самый разрыв, который трюк НЕ пропустил
               в непрерывный ряд.
    """
    series: List[dict] = []
    rolls: List[dict] = []
    k = k0
    prev_close: Optional[float] = None
    prev_ticker: Optional[str] = None

    bounds = []  # (start_date, end_date_exclusive, ticker)
    for i, (d, t) in enumerate(schedule):
        end = schedule[i + 1][0] if i + 1 < len(schedule) else date.max
        bounds.append((d, end, t))

    for start, end, ticker in bounds:
        chunk = [c for c in candles_by_ticker.get(ticker, []) if start <= c[0].date() < end]
        if not chunk:
            continue
        for j, (t, close, vol) in enumerate(chunk):
            if j == 0 and prev_close is not None:
                # первый бар нового контракта: δ считается от ЕГО открытия,
                # т.е. разрыв между контрактами в NAV не попадает
                gap_pct = (close / prev_close - 1.0) * 100.0
                rolls.append({
                    "date": t.date(), "from": prev_ticker, "to": ticker,
                    "price_from": prev_close, "price_to": close,
                    "gap_pct": gap_pct, "nav": k,
                })
                if roll_cost_bps:
                    k *= (1.0 - roll_cost_bps / 10_000.0)
                delta_ret = 0.0
                is_roll = True
            elif j == 0:
                delta_ret = 0.0
                is_roll = False
            else:
                delta_ret = close / chunk[j - 1][1] - 1.0
                is_roll = False
            k *= (1.0 + delta_ret)
            series.append({
                "time": t, "ticker": ticker, "raw_close": close, "nav": k,
                "volume": vol, "dollar_volume": close * vol, "is_roll": is_roll,
            })
            prev_close = close
            prev_ticker = ticker
    return series, rolls
