"""
Диагностика перед решением "склейка vs пул контрактов" (см. CLAUDE.md,
открытый вопрос про архитектуру моделирования).

Только чтение из БД, ничего не пишет и не меняет. Пороги ниже — явные
константы, а не "на глаз": вывод в конце вычисляется из тех же чисел, что
напечатаны в теле отчёта, чтобы результат можно было проверить и
воспроизвести (в том числе для статьи).

Запуск:
    .venv/bin/python -m scripts.diagnose_roll_splice --root CNYRUBF
    .venv/bin/python -m scripts.diagnose_roll_splice --root BR --near-bucket-days 30 90 180
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from db.database import engine

# --- Пороги диагностики (явно, чтобы вывод был воспроизводим и проверяем) ---

# Скачок на роллах считаем "значимым", если он больше, чем во столько раз
# превышает обычный дневной диапазон контракта. 2x выбрано как консервативная
# граница: скачок такого масштаба выделяется на фоне обычных дневных
# движений сильнее, чем типичный шум.
GAP_SIGNIFICANT_RATIO = 2.0

# Если значимые скачки (см. выше) встречаются чаще, чем в этой доле
# переходов — считаем наивную склейку рискованной.
GAP_SIGNIFICANT_SHARE_WARNING = 0.2  # 20%

# Горизонт "ближний контракт" для оценки концентрации объёма — сколько дней
# до экспирации считаем "тем самым активно торгуемым контрактом".
POOL_SAFETY_HORIZON_DAYS = 90

# Если внутри этого горизонта сосредоточена такая доля всего объёма — пул
# контрактов считаем не теряющим существенного покрытия сигнала.
POOL_SAFETY_VOLUME_SHARE = 0.8  # 80%


def get_contract_chain(root_symbol: str):
    """Все контракты серии по порядку экспирации."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT figi, ticker, expiration_date
            FROM futures_contracts
            WHERE root_symbol = :root
            ORDER BY expiration_date
        """), {"root": root_symbol}).fetchall()
    return rows


def get_first_last_candle(figi: str):
    with engine.connect() as conn:
        first = conn.execute(text("""
            SELECT time, close FROM futures_candles
            WHERE figi = :figi ORDER BY time ASC LIMIT 1
        """), {"figi": figi}).fetchone()
        last = conn.execute(text("""
            SELECT time, close FROM futures_candles
            WHERE figi = :figi ORDER BY time DESC LIMIT 1
        """), {"figi": figi}).fetchone()
    return first, last


def get_close_on_or_after(figi: str, day):
    """Первая доступная свеча в день `day` или позже (последняя свеча этого дня)."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT time, close FROM futures_candles
            WHERE figi = :figi AND time::date >= :day
            ORDER BY time ASC LIMIT 1
        """), {"figi": figi, "day": day}).fetchone()
        if row is None:
            return None
        day_found = row[0].date()
        last_of_day = conn.execute(text("""
            SELECT time, close FROM futures_candles
            WHERE figi = :figi AND time::date = :day_found
            ORDER BY time DESC LIMIT 1
        """), {"figi": figi, "day_found": day_found}).fetchone()
    return last_of_day


def get_avg_daily_range_pct(figi: str):
    """
    Средний внутридневной диапазон (high-low)/close в % — эталон "обычного"
    дневного движения, с которым сравниваем скачок на стыке контрактов.
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT avg((day_high - day_low) / NULLIF(day_close, 0) * 100) AS avg_range_pct,
                   count(*) AS days
            FROM (
                SELECT time::date AS d,
                       max(high) AS day_high,
                       min(low) AS day_low,
                       (array_agg(close ORDER BY time DESC))[1] AS day_close
                FROM futures_candles
                WHERE figi = :figi
                GROUP BY time::date
            ) daily
        """), {"figi": figi}).fetchone()
    return row.avg_range_pct, row.days


def diagnostic_1_roll_gaps(root_symbol: str) -> dict:
    print(f"\n=== Диагностика 1: скачок цены на стыке контрактов ({root_symbol}) ===")
    print(f"    Порог 'значимого' скачка: > {GAP_SIGNIFICANT_RATIO:.1f}x обычного дневного диапазона контракта.\n")
    chain = get_contract_chain(root_symbol)
    if len(chain) < 2:
        print("  Недостаточно контрактов для анализа переходов.")
        return {"transitions": 0}

    today = date.today()
    gaps = []
    ratios = []
    skipped_live = 0
    for near, next_ in zip(chain, chain[1:]):
        if near.expiration_date >= today:
            # Ближний контракт ещё не истёк — "последняя свеча" тут означает
            # просто "данные пока не загружены дальше", а не конец жизни
            # контракта. Это НЕ настоящий ролл, считать его как переход
            # нельзя — иначе в статистику попадёт снимок двух одновременно
            # живых контрактов вместо реального скачка на экспирации.
            skipped_live += 1
            continue

        near_first, near_last = get_first_last_candle(near.figi)
        if near_last is None:
            continue
        next_close_row = get_close_on_or_after(next_.figi, near_last.time.date())
        if next_close_row is None:
            print(f"  {near.ticker} -> {next_.ticker}: нет данных по следующему контракту, пропуск")
            continue

        near_close = float(near_last.close)
        next_close = float(next_close_row.close)
        gap_pct = (next_close - near_close) / near_close * 100

        avg_range_pct, days = get_avg_daily_range_pct(near.figi)
        avg_range_pct = float(avg_range_pct) if avg_range_pct else None
        ratio = (abs(gap_pct) / avg_range_pct) if avg_range_pct else None

        gaps.append(gap_pct)
        if ratio is not None:
            ratios.append(ratio)
        flag = " ⚠️ ЗНАЧИМЫЙ" if (ratio is not None and ratio > GAP_SIGNIFICANT_RATIO) else ""
        ratio_str = f"{ratio:.1f}x" if ratio is not None else "н/д"
        day_diff = (next_close_row.time.date() - near_last.time.date()).days
        print(f"  {near.ticker:<8} -> {next_.ticker:<8}  "
              f"roll {near_last.time.date()} -> {next_close_row.time.date()} (+{day_diff}д)  "
              f"close {near_close:.4f} -> {next_close:.4f}  "
              f"скачок {gap_pct:+.3f}%  "
              f"(обычный дневной range ~{avg_range_pct:.3f}% за {days} дн., скачок = {ratio_str} от него){flag}")

    result = {"transitions": len(gaps), "skipped_live": skipped_live}
    if gaps:
        abs_gaps = [abs(g) for g in gaps]
        significant = [r for r in ratios if r > GAP_SIGNIFICANT_RATIO]
        significant_share = len(significant) / len(ratios) if ratios else 0.0
        result.update({
            "avg_abs_gap_pct": sum(abs_gaps) / len(abs_gaps),
            "max_abs_gap_pct": max(abs_gaps),
            "significant_count": len(significant),
            "significant_share": significant_share,
        })
        print(f"\n  Итог по {len(gaps)} реальным переходам: средний |скачок| = {result['avg_abs_gap_pct']:.3f}%, "
              f"максимальный = {result['max_abs_gap_pct']:.3f}%")
        print(f"  Значимых скачков (> {GAP_SIGNIFICANT_RATIO:.1f}x обычной волатильности): "
              f"{len(significant)} из {len(ratios)} ({significant_share:.0%})")
    if skipped_live:
        print(f"  ({skipped_live} пар пропущено — ближний контракт ещё не истёк на сегодня, "
              f"это не настоящий ролл)")

    return result


def diagnostic_2_volume_by_days_to_expiry(root_symbol: str, buckets) -> dict:
    print(f"\n=== Диагностика 2: концентрация объёма по времени до экспирации ({root_symbol}) ===")
    print(f"    Горизонт 'ближнего контракта': последние {POOL_SAFETY_HORIZON_DAYS} дн. до экспирации.\n")
    buckets = sorted(buckets)
    case_lines = []
    prev = None
    for b in buckets:
        if prev is None:
            case_lines.append(f"WHEN (fc.expiration_date - fcd.time::date) <= {b} THEN '<= {b}д'")
        else:
            case_lines.append(f"WHEN (fc.expiration_date - fcd.time::date) <= {b} THEN '{prev}-{b}д'")
        prev = b
    case_lines.append(f"ELSE '> {prev}д'")
    case_sql = "CASE " + " ".join(case_lines) + " END"

    query = text(f"""
        SELECT {case_sql} AS bucket,
               sum(fcd.volume) AS volume,
               count(*) AS candles
        FROM futures_candles fcd
        JOIN futures_contracts fc USING (figi)
        WHERE fc.root_symbol = :root
        GROUP BY bucket
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"root": root_symbol}).fetchall()
        near_row = conn.execute(text("""
            SELECT sum(fcd.volume) AS near_volume
            FROM futures_candles fcd
            JOIN futures_contracts fc USING (figi)
            WHERE fc.root_symbol = :root
              AND (fc.expiration_date - fcd.time::date) <= :horizon
        """), {"root": root_symbol, "horizon": POOL_SAFETY_HORIZON_DAYS}).fetchone()

    total_volume = sum(r.volume for r in rows) or 1
    # Сортируем по смыслу: сначала "далеко от экспирации" (самый большой
    # бакет), потом "близко". У самого дальнего бакета (> buckets[-1]) должен
    # быть САМЫЙ БОЛЬШОЙ ключ, чтобы при sorted(..., reverse=True) он встал
    # первым.
    order_key = {f"> {buckets[-1]}д": len(buckets)}
    for i, b in enumerate(buckets):
        label = f"<= {b}д" if i == 0 else f"{buckets[i-1]}-{b}д"
        order_key[label] = i
    rows_sorted = sorted(rows, key=lambda r: order_key.get(r.bucket, 999), reverse=True)

    for r in rows_sorted:
        share = r.volume / total_volume * 100
        bar = "█" * int(share / 2)
        print(f"  {r.bucket:<12} объём {r.volume:>15,}  ({share:5.1f}%)  {bar}")

    near_share = (near_row.near_volume or 0) / total_volume
    print(f"\n  Доля объёма в последние {POOL_SAFETY_HORIZON_DAYS} дней перед экспирацией: {near_share:.1%} "
          f"(порог безопасности пула: {POOL_SAFETY_VOLUME_SHARE:.0%})")

    return {"near_share": near_share, "horizon_days": POOL_SAFETY_HORIZON_DAYS}


def print_verdict(root_symbol: str, d1: dict, d2: dict):
    print(f"\n=== Вывод ({root_symbol}) ===")
    print(f"    Критерии: доля значимых скачков на роллах <= {GAP_SIGNIFICANT_SHARE_WARNING:.0%} "
          f"И доля объёма в последние {POOL_SAFETY_HORIZON_DAYS} дн. >= {POOL_SAFETY_VOLUME_SHARE:.0%}\n")

    if d1.get("transitions", 0) == 0:
        print("  Недостаточно данных о переходах для вывода.")
        return

    significant_share = d1.get("significant_share", 0.0)
    near_share = d2.get("near_share", 0.0)

    gap_risky = significant_share > GAP_SIGNIFICANT_SHARE_WARNING
    volume_safe_for_pool = near_share >= POOL_SAFETY_VOLUME_SHARE

    print(f"  Скачки на роллах: {significant_share:.0%} значимых "
          f"({'выше' if gap_risky else 'не выше'} порога {GAP_SIGNIFICANT_SHARE_WARNING:.0%}) "
          f"— {'риск для наивной склейки' if gap_risky else 'наивная склейка выглядит приемлемой по этому критерию'}.")
    print(f"  Концентрация объёма: {near_share:.0%} в последние {POOL_SAFETY_HORIZON_DAYS} дн. "
          f"({'>=' if volume_safe_for_pool else '<'} порога {POOL_SAFETY_VOLUME_SHARE:.0%}) "
          f"— {'пул почти не теряет покрытия сигнала' if volume_safe_for_pool else 'пул может терять заметную часть сигнала из дальних контрактов'}.")

    if gap_risky and volume_safe_for_pool:
        print(f"\n  ➜ ПУЛ КОНТРАКТОВ выглядит безопаснее наивной склейки для {root_symbol}: "
              f"скачки на роллах существенны, а объём и так сосредоточен в ближних контрактах.")
    elif not gap_risky:
        print(f"\n  ➜ Скачки на роллах в пределах нормы — риск наивной склейки для {root_symbol} невысок "
              f"по этому критерию, но потребует ratio/back-adjustment для методологической чистоты.")
    elif gap_risky and not volume_safe_for_pool:
        print(f"\n  ➜ Скачки значимы, но объём НЕ так сильно сконцентрирован в ближних контрактах — "
              f"пул может терять сигнал из дальних контрактов. Нужен третий вариант "
              f"(back-adjustment/ratio-adjustment) вместо простого выбора между пулом и наивной склейкой.")

    print("\n  ⚠️ Это эвристическая оценка по фиксированным порогам на одной серии, а не финальное "
          "архитектурное решение — см. открытый вопрос в plan.md.")


def main():
    parser = argparse.ArgumentParser(description="Диагностика склейки/пула фьючерсных контрактов")
    parser.add_argument("--root", required=True, help="root_symbol серии, например CNYRUBF")
    parser.add_argument("--near-bucket-days", type=int, nargs="+", default=[30, 90, 180],
                         help="границы бакетов 'дней до экспирации' для диагностики 2 (по возрастанию)")
    args = parser.parse_args()

    d1 = diagnostic_1_roll_gaps(args.root)
    d2 = diagnostic_2_volume_by_days_to_expiry(args.root, args.near_bucket_days)
    print_verdict(args.root, d1, d2)


if __name__ == "__main__":
    main()
