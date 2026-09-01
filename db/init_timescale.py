import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from db.database import engine, Base
from db.models import FuturesRoot, FuturesContract


def init_timescale():
    """
    Инициализирует расширение TimescaleDB в базе futures_lab, создаёт
    гипертаблицу futures_candles и continuous aggregates для M15/H1/1D —
    аналог ../agent/db/init_timescale.py, но для фьючерсов.
    """
    print("🚀 1. Инициализация базы данных и расширений...")
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))

        print("🧹 2. Удаление старых представлений и таблиц...")
        conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS futures_candles_1d CASCADE;"))
        conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS futures_candles_h1 CASCADE;"))
        conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS futures_candles_m15 CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS futures_candles_1d CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS futures_candles_h1 CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS futures_candles_m15 CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS futures_candles CASCADE;"))

    print("📦 3. Создание базовых таблиц (futures_roots, futures_contracts)...")
    Base.metadata.create_all(bind=engine, tables=[FuturesRoot.__table__, FuturesContract.__table__])

    print("📊 4. Создание гипертаблицы futures_candles (1M свечи)...")
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS futures_candles (
                time TIMESTAMPTZ NOT NULL,
                figi VARCHAR(64) NOT NULL REFERENCES futures_contracts(figi) ON DELETE CASCADE,
                open NUMERIC(18, 9) NOT NULL,
                high NUMERIC(18, 9) NOT NULL,
                low NUMERIC(18, 9) NOT NULL,
                close NUMERIC(18, 9) NOT NULL,
                volume BIGINT NOT NULL,
                candle_source SMALLINT,
                ingest_source VARCHAR(16),
                PRIMARY KEY (time, figi)
            );
        """))

        conn.execute(text("""
            SELECT create_hypertable('futures_candles', by_range('time'), if_not_exists => TRUE);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_futures_candles_figi_time ON futures_candles (figi, time DESC);
        """))

        print("📈 5. Создание Continuous Aggregates (M15, H1, 1D)...")

        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS futures_candles_m15
            WITH (timescaledb.continuous) AS
            SELECT figi,
                   time_bucket(INTERVAL '15 minutes', time) AS time,
                   first(open, time)  AS open,
                   max(high)          AS high,
                   min(low)           AS low,
                   last(close, time)  AS close,
                   sum(volume)        AS volume
            FROM futures_candles
            GROUP BY figi, time_bucket(INTERVAL '15 minutes', time)
            WITH NO DATA;
        """))

        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS futures_candles_h1
            WITH (timescaledb.continuous) AS
            SELECT figi,
                   time_bucket(INTERVAL '1 hour', time) AS time,
                   first(open, time)  AS open,
                   max(high)          AS high,
                   min(low)           AS low,
                   last(close, time)  AS close,
                   sum(volume)        AS volume
            FROM futures_candles
            GROUP BY figi, time_bucket(INTERVAL '1 hour', time)
            WITH NO DATA;
        """))

        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS futures_candles_1d
            WITH (timescaledb.continuous) AS
            SELECT figi,
                   time_bucket(INTERVAL '1 day', time) AS time,
                   first(open, time)  AS open,
                   max(high)          AS high,
                   min(low)           AS low,
                   last(close, time)  AS close,
                   sum(volume)        AS volume
            FROM futures_candles
            GROUP BY figi, time_bucket(INTERVAL '1 day', time)
            WITH NO DATA;
        """))

        print("⚙️ 6. Настройка политик автообновления витрин...")
        conn.execute(text("""
            SELECT add_continuous_aggregate_policy('futures_candles_m15',
                start_offset => INTERVAL '3 days',
                end_offset => INTERVAL '1 minute',
                schedule_interval => INTERVAL '5 minutes',
                if_not_exists => TRUE);
        """))

        conn.execute(text("""
            SELECT add_continuous_aggregate_policy('futures_candles_h1',
                start_offset => INTERVAL '14 days',
                end_offset => INTERVAL '5 minutes',
                schedule_interval => INTERVAL '15 minutes',
                if_not_exists => TRUE);
        """))

        conn.execute(text("""
            SELECT add_continuous_aggregate_policy('futures_candles_1d',
                start_offset => INTERVAL '30 days',
                end_offset => INTERVAL '1 hour',
                schedule_interval => INTERVAL '1 hour',
                if_not_exists => TRUE);
        """))

        print("🗜️ 7. Настройка политики колоночного сжатия (Compression)...")
        conn.execute(text("""
            ALTER TABLE futures_candles SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'figi',
                timescaledb.compress_orderby = 'time ASC'
            );
        """))
        conn.execute(text("""
            SELECT add_compression_policy('futures_candles', INTERVAL '7 days', if_not_exists => TRUE);
        """))

    print("✅ Инициализация TimescaleDB (futures_lab) успешно завершена!")


if __name__ == "__main__":
    init_timescale()
