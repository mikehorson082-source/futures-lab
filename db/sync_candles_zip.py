import io
import os
import sys
import time
import zipfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import text

from db.database import engine, get_db_session
from db.models import FuturesContract

load_dotenv()
TINKOFF_TOKEN = os.getenv("TINKOFF_SANDBOX_TOKEN") or os.getenv("TINKOFF_TOKEN")


def download_candles_zip(figi: str, year: int, token: str, max_retries: int = 3, timeout_sec: int = 120) -> Optional[bytes]:
    """
    Скачивает ZIP-архив минутных свечей за указанный год через T-Bank History REST API.
    Идентично ../agent/db/sync_candles_zip.py — эндпоинт один и тот же для любого figi,
    акция это или фьючерс.
    """
    url = f"https://invest-public-api.tbank.ru/history-data?figi={figi}&year={year}"
    cmd = [
        "curl", "-4", "-k", "-s",
        "-w", "\n%{http_code}",
        "-H", f"Authorization: Bearer {token}",
        url
    ]

    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
            output = result.stdout
            if not output:
                continue

            parts = output.rsplit(b"\n", 1)
            if len(parts) != 2:
                continue

            body, code_bytes = parts
            http_code = code_bytes.decode("utf-8", errors="ignore").strip()

            if http_code == "200" and len(body) > 100:
                if zipfile.is_zipfile(io.BytesIO(body)):
                    return body
                else:
                    print(f"    ⚠️ Архив для {figi} ({year} г.) повреждён или загружен не полностью. Повтор {attempt}/{max_retries}...", flush=True)
                    time.sleep(3)
            elif http_code == "429":
                print(f"    ⚠️ Rate limit (429)! Пауза 6 секунд (попытка {attempt}/{max_retries})...", flush=True)
                time.sleep(6)
            elif http_code == "404" or http_code == "500":
                # 404/500 обычно означает, что контракт ещё не существовал / уже
                # истёк до начала этого года — для фьючерса это нормальная
                # ситуация чаще, чем для акции: у каждого контракта всего
                # несколько месяцев жизни.
                return None
            else:
                time.sleep(2)
        except subprocess.TimeoutExpired:
            print(f"    ⏳ Таймаут ({timeout_sec}с) для {figi} ({year} г.). Попытка {attempt}/{max_retries}...", flush=True)
            time.sleep(3)
        except Exception as e:
            print(f"    ❌ Ошибка загрузки архива ({figi}, {year}): {e}", flush=True)
            time.sleep(2)

    return None


def parse_candles_csv(zip_bytes: bytes, figi: str) -> List[tuple]:
    """
    Распаковывает CSV из ZIP в памяти и парсит минутные свечи.
    Формат CSV: uid;timestamp;open;close;high;low;volume;
    Проставляет candle_source=1 (EXCHANGE) и ingest_source='zip'.
    """
    candles = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for filename in zf.namelist():
                if not filename.endswith(".csv"):
                    continue
                with zf.open(filename) as f:
                    for line_bytes in f:
                        line = line_bytes.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        parts = line.split(";")
                        if len(parts) < 7:
                            continue

                        time_str = parts[1]
                        try:
                            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                            open_p = float(parts[2])
                            close_p = float(parts[3])
                            high_p = float(parts[4])
                            low_p = float(parts[5])
                            vol = int(parts[6])

                            candles.append((dt, figi, open_p, high_p, low_p, close_p, vol, 1, 'zip'))
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        print(f"❌ Ошибка парсинга архива для {figi}: {e}")
    return candles


def insert_candles_batch(candles: List[tuple]):
    """Быстрая вставка пачки минутных свечей через execute_values."""
    if not candles:
        return 0

    query = """
    INSERT INTO futures_candles (time, figi, open, high, low, close, volume, candle_source, ingest_source)
    VALUES %s
    ON CONFLICT (time, figi) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        candle_source = EXCLUDED.candle_source,
        ingest_source = EXCLUDED.ingest_source;
    """

    db_raw = engine.raw_connection()
    try:
        with db_raw.cursor() as cursor:
            execute_values(cursor, query, candles, page_size=10000)
        db_raw.commit()
        return len(candles)
    except Exception as e:
        db_raw.rollback()
        print(f"❌ Ошибка вставки в БД: {e}")
        return 0
    finally:
        db_raw.close()


def is_year_already_loaded(db, figi: str, year: int) -> bool:
    start_dt = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    stmt = text("SELECT 1 FROM futures_candles WHERE figi = :figi AND time >= :start_dt AND time < :end_dt LIMIT 1")
    res = db.execute(stmt, {"figi": figi, "start_dt": start_dt, "end_dt": end_dt}).scalar()
    return res is not None


def sync_contract_history_zip(contract: FuturesContract, years: List[int], token: str = TINKOFF_TOKEN, skip_existing: bool = True) -> int:
    """Синхронизирует архивную историю минутных свечей для одного контракта по указанным годам."""
    total_candles = 0
    db = get_db_session()

    try:
        for year in years:
            if skip_existing and is_year_already_loaded(db, contract.figi, year):
                print(f"    ℹ️ [{contract.ticker}] {year} год уже загружен в БД, пропускаем...", flush=True)
                continue

            zip_data = download_candles_zip(contract.figi, year, token)
            if not zip_data:
                continue

            parsed_candles = parse_candles_csv(zip_data, contract.figi)
            if parsed_candles:
                count = insert_candles_batch(parsed_candles)
                total_candles += count
                print(f"    ✅ [{contract.ticker}] {year} год: сохранено {count:,} свечей 1M.", flush=True)

            time.sleep(1.0)  # лимит T-Bank History API — до 30 запросов в минуту

    finally:
        db.close()

    return total_candles


def sync_all_contracts_zip(
    root_symbols: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    skip_existing: bool = True,
):
    """
    Синхронизирует исторические минутные свечи из ZIP-архивов для всех
    контрактов из futures_contracts (или только для указанных root-серий).
    """
    if years is None:
        # 2022-04-01 — граница режимного сдвига, единая для всего проекта (см. CLAUDE.md)
        years = [2022, 2023, 2024, 2025, 2026]

    db = get_db_session()
    try:
        query = db.query(FuturesContract)
        if root_symbols:
            query = query.filter(FuturesContract.root_symbol.in_(root_symbols))
        contracts = query.order_by(FuturesContract.root_symbol, FuturesContract.expiration_date).all()

        print(f"🚀 Старт загрузки архивов 1M для {len(contracts)} контрактов, годы-кандидаты {years}...", flush=True)
        total_all = 0
        skipped_no_overlap = 0

        for idx, contract in enumerate(contracts, 1):
            # Контракт мог существовать только в пределах своей реальной жизни —
            # от начала торгов (обычно за несколько месяцев до экспирации) до
            # даты экспирации. Раньше здесь перебирались все годы из `years`
            # для каждого контракта вслепую: контракт, истёкший в 2021-м,
            # получал 5 заведомо бесполезных запросов к API вместо 0.
            start_year = contract.first_trade_date.year if contract.first_trade_date else contract.expiration_date.year - 1
            relevant_years = [y for y in years if start_year <= y <= contract.expiration_date.year]

            if not relevant_years:
                skipped_no_overlap += 1
                print(f"[{idx}/{len(contracts)}] {contract.root_symbol} / {contract.ticker} (эксп. {contract.expiration_date}) — "
                      f"нет пересечения с {years}, пропуск без запросов к API", flush=True)
                continue

            print(f"[{idx}/{len(contracts)}] {contract.root_symbol} / {contract.ticker} "
                  f"(эксп. {contract.expiration_date}, годы {relevant_years})...", flush=True)
            count = sync_contract_history_zip(contract, years=relevant_years, skip_existing=skip_existing)
            total_all += count

        if skipped_no_overlap:
            print(f"    ({skipped_no_overlap} контрактов пропущено без обращения к API — их жизнь не пересекается с {years})", flush=True)

        print(f"🎉 Загрузка архивов завершена! Всего сохранено свечей 1M: {total_all:,}", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    sync_all_contracts_zip()
