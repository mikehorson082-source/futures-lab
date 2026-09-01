from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Numeric,
    Boolean,
    BigInteger,
    SmallInteger,
    DateTime,
    Date,
    Text,
    ForeignKey,
    PrimaryKeyConstraint,
)
from sqlalchemy.orm import relationship
from .database import Base


class FuturesRoot(Base):
    """
    Root-серия фьючерсов — группировка по базовому активу, а не конкретный
    контракт. Например, 'CNYRUBF' объединяет все контракты CRU6, CRZ6, CRH7...
    См. CLAUDE.md: хранение контрактов раздельно, склейка — отдельный шаг позже.
    """

    __tablename__ = "futures_roots"

    root_symbol = Column(String(32), primary_key=True)          # 'SBERF', 'CNYRUBF', 'IMOEXF', 'BR'
    basic_asset = Column(String(64), nullable=False, index=True)  # точное значение из API: 'SBER', 'CNY/RUB', 'IMOEX', 'Brent'
    underlying_type = Column(String(16), nullable=False)         # 'share' | 'currency' | 'index' | 'commodity'
    underlying_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    contracts = relationship("FuturesContract", back_populates="root", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<FuturesRoot(root_symbol='{self.root_symbol}', basic_asset='{self.basic_asset}')>"


class FuturesContract(Base):
    """
    Один конкретный фьючерсный контракт (например, CRU6 с экспирацией 2026-09-18).
    Аналог Share из ../agent, но с обязательной датой экспирации — ключевое
    отличие фьючерса от акции как объекта в БД.
    """

    __tablename__ = "futures_contracts"

    figi = Column(String(64), primary_key=True)
    root_symbol = Column(String(32), ForeignKey("futures_roots.root_symbol"), nullable=False, index=True)
    ticker = Column(String(32), nullable=False, index=True)      # 'CRU6'
    name = Column(String(255), nullable=False)
    expiration_date = Column(Date, nullable=False, index=True)
    first_trade_date = Column(Date, nullable=True)
    last_trade_date = Column(Date, nullable=True)
    currency = Column(String(16), nullable=False)
    lot = Column(Integer, nullable=False)
    min_price_increment = Column(Numeric(18, 9), nullable=True)
    api_trade_available_flag = Column(Boolean, default=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    root = relationship("FuturesRoot", back_populates="contracts")
    candles = relationship("FuturesCandle", back_populates="contract", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<FuturesContract(ticker='{self.ticker}', figi='{self.figi}', exp='{self.expiration_date}')>"


class FuturesCandle(Base):
    """
    Основная гипертаблица минутных свечей (1M) в TimescaleDB для фьючерсов.
    Единственный источник правды для всех старших таймфреймов (аналог
    candles из ../agent).
    """

    __tablename__ = "futures_candles"

    time = Column(DateTime(timezone=True), nullable=False)
    figi = Column(String(64), ForeignKey("futures_contracts.figi"), nullable=False)
    open = Column(Numeric(18, 9), nullable=False)
    high = Column(Numeric(18, 9), nullable=False)
    low = Column(Numeric(18, 9), nullable=False)
    close = Column(Numeric(18, 9), nullable=False)
    volume = Column(BigInteger, nullable=False)
    # Источник свечи по бирже, из t_tech.invest.schemas.CandleSource
    # (1=EXCHANGE, 2=DEALER_WEEKEND, 3=INCLUDE_WEEKEND). Для строк из ZIP
    # проставляется вручную как EXCHANGE (см. ingest_source ниже и
    # plan.md, раздел 7.4 — решение принято после выборочной проверки
    # через gRPC, не построчно).
    candle_source = Column(SmallInteger, nullable=True)
    # Какой наш загрузчик записал строку: 'zip' (db/sync_candles_zip.py,
    # архивы history-data) или 'grpc' (будущая догрузка через
    # GetCandles). Не про биржу, а про наш пайплайн — чтобы всегда можно
    # было понять, откуда взялась конкретная строка.
    ingest_source = Column(String(16), nullable=True)

    __table_args__ = (PrimaryKeyConstraint("time", "figi"),)

    contract = relationship("FuturesContract", back_populates="candles")

    def __repr__(self):
        return f"<FuturesCandle(figi='{self.figi}', time='{self.time}', close={self.close})>"


class FuturesCandleM15(Base):
    """TimescaleDB Continuous Aggregate витрина: таймфрейм 15 минут."""

    __tablename__ = "futures_candles_m15"

    time = Column(DateTime(timezone=True), primary_key=True)
    figi = Column(String(64), ForeignKey("futures_contracts.figi"), primary_key=True)
    open = Column(Numeric(18, 9), nullable=False)
    high = Column(Numeric(18, 9), nullable=False)
    low = Column(Numeric(18, 9), nullable=False)
    close = Column(Numeric(18, 9), nullable=False)
    volume = Column(BigInteger, nullable=False)


class FuturesCandleH1(Base):
    """TimescaleDB Continuous Aggregate витрина: таймфрейм 1 час."""

    __tablename__ = "futures_candles_h1"

    time = Column(DateTime(timezone=True), primary_key=True)
    figi = Column(String(64), ForeignKey("futures_contracts.figi"), primary_key=True)
    open = Column(Numeric(18, 9), nullable=False)
    high = Column(Numeric(18, 9), nullable=False)
    low = Column(Numeric(18, 9), nullable=False)
    close = Column(Numeric(18, 9), nullable=False)
    volume = Column(BigInteger, nullable=False)


class TradingSchedule(Base):
    """
    Календарь торговых дней FORTS — по свечам, а не по расписанию биржи.
    См. CLAUDE.md / plan.md (раздел про календарь): ни ISS `dailytable`,
    ни T-Invest `trading_schedules` не дают полной и надёжной истории для
    срочного рынка (первый неполон, второй не смотрит в прошлое вообще) —
    поэтому источник истины здесь — фактическое наличие свечей, а
    задокументированные даты смены режима биржи используются как
    проверка результата, а не как исходные данные (см. db/sync_calendar.py).
    """

    __tablename__ = "trading_schedules"

    date = Column(Date, primary_key=True)
    weekday = Column(Integer, nullable=False)  # 0=понедельник ... 6=воскресенье (python date.weekday())
    is_trading_day = Column(Boolean, nullable=False)  # была ли хоть одна свеча в этот день хоть в одной из отслеживаемых серий
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<TradingSchedule(date='{self.date}', is_trading_day={self.is_trading_day})>"


class FuturesCandle1D(Base):
    """TimescaleDB Continuous Aggregate витрина: таймфрейм 1 день."""

    __tablename__ = "futures_candles_1d"

    time = Column(DateTime(timezone=True), primary_key=True)
    figi = Column(String(64), ForeignKey("futures_contracts.figi"), primary_key=True)
    open = Column(Numeric(18, 9), nullable=False)
    high = Column(Numeric(18, 9), nullable=False)
    low = Column(Numeric(18, 9), nullable=False)
    close = Column(Numeric(18, 9), nullable=False)
    volume = Column(BigInteger, nullable=False)
