"""
Календарь торговых дней FORTS.

Источник истины — фактическое наличие свечей в futures_candles, не
расписание биржи. Причина зафиксирована в CLAUDE.md/plan.md: для срочного
рынка не существует источника, который надёжно и полно знает прошлое —
* T-Invest `trading_schedules` вообще не смотрит назад ("`from` can't be
  less than the current date");
* T-Invest `session_schedule` (через ISS `engines/futures.json`) — тоже
  только "сегодня", параметр `date=` игнорируется;
* ISS `dailytable` (тот же эндпоинт) историчен, но неполон — например,
  пропускает нерабочую субботу 2026-08-01, которую свечи ловят правильно.

Поэтому здесь `dailytable` и три задокументированные даты смены режима
биржи используются только как ПРОВЕРКА результата (assert), а не как
источник данных для его построения.

Запуск:
    .venv/bin/python -m db.sync_calendar
"""
import subprocess
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from deprecation import DeprecatedWarning
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.database import engine, Base, get_db_session
from db.models import TradingSchedule

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecatedWarning)

# ISS-эндпоинт срочного рынка целиком (engine-уровень, не по инструментам).
# `dailytable` внутри — список ИСКЛЮЧЕНИЙ (праздники/рабочие субботы) с 2018
# года, используется только для сверки, см. докстринг файла.
ISS_ENGINE_URL = "https://iss.moex.com/iss/engines/futures.json"

# Даты смены режима биржи, найденные и проверенные в разговоре с
# пользователем (см. plan.md) — используются как assert, не как источник
# времени сессий (режим меняется слишком часто, чтобы на него полагаться
# как на константу, см. plan.md).
DSVD_LAUNCH_DATE = date(2025, 8, 16)          # запуск сессии выходного дня (не для валюты)
DSVD_CURRENCY_DATE = date(2026, 7, 18)        # валютные фьючерсы допущены к сессии выходного дня
CNYRUBF_ROOTS = {"CNYRUBF", "CNYRUBF_PERP"}   # серии, подпадающие под второе правило


def _fetch_iss_dailytable() -> dict[date, dict]:
    """
    Список исключений срочного рынка из ISS (праздники/рабочие субботы).
    Через curl -4: прямой urllib здесь стабильно уходит в таймаут в этом
    окружении (та же особенность, что у sync_candles_zip.py).
    """
    cmd = ["curl", "-4", "-sS", "-m", "20", ISS_ENGINE_URL]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=25)
        data = json.loads(result.stdout.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Не удалось получить dailytable из ISS ({e}) — сверка будет пропущена.")
        return {}

    exceptions = {}
    for row in data.get("dailytable", {}).get("data", []):
        d = date.fromisoformat(row[0])
        exceptions[d] = {"is_work_day": row[1], "start": row[2], "stop": row[3]}
    return exceptions


def _candle_data_date_range(db) -> tuple[Optional[date], Optional[date]]:
    row = db.execute(text("SELECT MIN(time)::date, MAX(time)::date FROM futures_candles")).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _trading_days_from_candles(db) -> set[date]:
    """Даты, где есть хотя бы одна свеча хоть в одной из отслеживаемых серий."""
    rows = db.execute(text("SELECT DISTINCT time::date AS d FROM futures_candles")).fetchall()
    return {r[0] for r in rows}


def _weekend_trading_dates_by_root(db, root_symbols: set[str]) -> set[date]:
    """Даты по выходным (сб/вс), где есть свечи у указанных root-серий."""
    rows = db.execute(
        text(
            """
            SELECT DISTINCT fc.time::date AS d
            FROM futures_candles fc
            JOIN futures_contracts ct ON ct.figi = fc.figi
            WHERE ct.root_symbol = ANY(:roots)
            """
        ),
        {"roots": list(root_symbols)},
    ).fetchall()
    return {r[0] for r in rows if r[0].weekday() >= 5}


def _run_sanity_checks(db, exceptions: dict[date, dict]) -> bool:
    """
    Проверка построенного календаря против задокументированных фактов о
    смене режима биржи. Не источник данных, а тест поверх результата —
    см. докстринг файла. Возвращает True, если всё сошлось.

    Официальные "рабочие субботы" (перенос выходного дня — механизм,
    существующий независимо от ДСВД и намного раньше её) исключаются явно
    через dailytable (is_work_day=1) — иначе они ложно выглядели бы как
    нарушение правила "до ДСВД по выходным не торговали".
    """
    print("\n🔎 Проверка календаря против задокументированных дат смены режима...")
    ok = True

    def _is_known_working_weekend(d: date) -> bool:
        exc = exceptions.get(d)
        return bool(exc and exc["is_work_day"] == 1)

    # 1. До запуска ДСВД (16.08.2025) выходных свечей быть не должно нигде,
    #    кроме уже известных официальных рабочих суббот (перенос выходного).
    all_roots = {r[0] for r in db.execute(text("SELECT DISTINCT root_symbol FROM futures_roots")).fetchall()}
    early_weekend = _weekend_trading_dates_by_root(db, all_roots)
    early_weekend_before_launch = {
        d for d in early_weekend if d < DSVD_LAUNCH_DATE and not _is_known_working_weekend(d)
    }
    if early_weekend_before_launch:
        print(
            f"   ❌ Найдены свечи по выходным ДО запуска ДСВД ({DSVD_LAUNCH_DATE}): "
            f"{sorted(early_weekend_before_launch)[:5]}{'...' if len(early_weekend_before_launch) > 5 else ''}"
        )
        ok = False
    else:
        print(f"   ✅ Свечей по выходным до {DSVD_LAUNCH_DATE} не найдено — совпадает с датой запуска ДСВД.")

    # 2. До допуска валюты (18.07.2026) у CNYRUBF/CNYRUBF_PERP выходных свечей
    #    быть не должно, кроме тех же официальных рабочих суббот.
    cny_weekend = _weekend_trading_dates_by_root(db, CNYRUBF_ROOTS)
    cny_weekend_before = {
        d for d in cny_weekend if d < DSVD_CURRENCY_DATE and not _is_known_working_weekend(d)
    }
    if cny_weekend_before:
        print(
            f"   ❌ Найдены свечи CNYRUBF/CNYRUBF_PERP по выходным ДО допуска валюты ({DSVD_CURRENCY_DATE}): "
            f"{sorted(cny_weekend_before)[:5]}{'...' if len(cny_weekend_before) > 5 else ''}"
        )
        ok = False
    else:
        print(f"   ✅ Свечей CNYRUBF по выходным до {DSVD_CURRENCY_DATE} не найдено — совпадает с датой допуска.")

    return ok


def sync_trading_calendar():
    print("=" * 80)
    print("🚀 ПОСТРОЕНИЕ КАЛЕНДАРЯ ТОРГОВЫХ ДНЕЙ FORTS (ПО СВЕЧАМ)")
    print("=" * 80)

    Base.metadata.create_all(bind=engine, tables=[TradingSchedule.__table__])

    db = get_db_session()
    try:
        start_d, end_d = _candle_data_date_range(db)
        if start_d is None:
            raise RuntimeError("В futures_candles нет данных — календарь строить не из чего.")
        print(f"📅 Диапазон данных: {start_d} … {end_d}")

        print("📥 Загрузка фактических торговых дат из свечей...")
        trading_dates = _trading_days_from_candles(db)
        print(f"   Торговых дней по свечам: {len(trading_dates):,}")

        exceptions = _fetch_iss_dailytable()
        if exceptions:
            print(f"   Загружено {len(exceptions)} исключений из ISS dailytable (только для сверки, см. ниже).")

        # Запись календаря: каждый день диапазона получает is_trading_day по свечам.
        print("💾 Запись в trading_schedules...")
        records = []
        d = start_d
        while d <= end_d:
            records.append(
                {
                    "date": d,
                    "weekday": d.weekday(),
                    "is_trading_day": d in trading_dates,
                }
            )
            d += timedelta(days=1)

        with engine.begin() as conn:
            for rec in records:
                stmt = pg_insert(TradingSchedule.__table__).values(**rec)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["date"],
                    set_={"weekday": stmt.excluded.weekday, "is_trading_day": stmt.excluded.is_trading_day},
                )
                conn.execute(stmt)
        print(f"   ✅ Записано {len(records):,} дней.")

        # Сверка с dailytable — только отчёт, ничего не переписывает.
        if exceptions:
            print("\n📋 Расхождения с ISS dailytable (не меняют календарь, только для информации):")
            n_mismatch = 0
            for d, exc in sorted(exceptions.items()):
                if not (start_d <= d <= end_d):
                    continue
                expected_trading = not (exc["is_work_day"] == 0)
                actual_trading = d in trading_dates
                if expected_trading != actual_trading:
                    n_mismatch += 1
                    print(f"   {d}: dailytable is_work_day={exc['is_work_day']}, по свечам торговали={actual_trading}")
            if n_mismatch == 0:
                print("   Расхождений нет.")
            else:
                print(f"   Итого расхождений: {n_mismatch} (ожидаемо — dailytable неполон, см. докстринг файла).")

        ok = _run_sanity_checks(db, exceptions)

        print("\n📊 ИТОГИ:")
        n_trading = sum(1 for r in records if r["is_trading_day"])
        print(f"   • Всего дней: {len(records):,}, торговых: {n_trading:,} ({n_trading/len(records)*100:.1f}%)")
        print(f"   • Проверки против дат смены режима: {'✅ пройдены' if ok else '❌ ЕСТЬ РАСХОЖДЕНИЯ, см. выше'}")
        print("=" * 80)

    finally:
        db.close()


if __name__ == "__main__":
    sync_trading_calendar()
