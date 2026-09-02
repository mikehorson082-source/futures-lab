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
    В отличие от акции как объекта в БД, здесь обязательна дата экспирации —
    ключевое отличие фьючерса.
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
    Единственный источник правды для всех старших таймфреймов.
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


class CurrencyRate(Base):
    """
    Дневной спот-курс валютной пары с биржи (MOEX ISS), не фьючерс.
    Нужен для разбора базиса на Этапе 1 (CLAUDE.md) — базис считается
    относительно спота, а не относительно другого фьючерса.
    """

    __tablename__ = "currency_rates"

    date = Column(Date, primary_key=True)
    pair = Column(String(16), primary_key=True)   # 'CNYRUB', позже 'USDRUB' и т.п.
    source = Column(String(16), nullable=False)   # 'moex_iss' — источник ряда, см. db/sync_reference_data.py
    open = Column(Numeric(18, 6), nullable=True)
    high = Column(Numeric(18, 6), nullable=True)
    low = Column(Numeric(18, 6), nullable=True)
    close = Column(Numeric(18, 6), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<CurrencyRate(pair='{self.pair}', date='{self.date}', close={self.close})>"


class IndexPrice(Base):
    """
    Дневное значение биржевого ИНДЕКСА (MOEX ISS), не фьючерс и не валюта.

    Отдельная таблица, а не `currency_rates` с выдуманной "парой": индекс —
    это уровень, а не курс обмена, и складывать их в одну таблицу значило бы
    ради экономии одной сущности запутать смысл колонки `pair`.

    Нужен для базиса Этапа 3 (IMOEXF): фьючерс на индекс сравнивается со
    значением самого индекса ровно так же, как валютный фьючерс со спотом.
    """

    __tablename__ = "index_prices"

    date = Column(Date, primary_key=True)
    symbol = Column(String(16), primary_key=True)   # 'IMOEX'
    source = Column(String(16), nullable=False)     # 'moex_iss'
    open = Column(Numeric(18, 6), nullable=True)
    high = Column(Numeric(18, 6), nullable=True)
    low = Column(Numeric(18, 6), nullable=True)
    close = Column(Numeric(18, 6), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<IndexPrice(symbol='{self.symbol}', date='{self.date}', close={self.close})>"


class EquityPrice(Base):
    """
    Дневная цена АКЦИИ с MOEX ISS (борд TQBR) — спот для фьючерса на
    отдельную бумагу (Этап 4, SBERF).

    Отдельно от `index_prices` по той же причине, по которой индекс отделён
    от валюты: это разные сущности с разными источниками и разной
    справедливой ценой (у акции в неё входят дивиденды, у индекса —
    усреднённая дивидендная доходность, у валюты — вторая ставка).
    """

    __tablename__ = "equity_prices"

    date = Column(Date, primary_key=True)
    ticker = Column(String(16), primary_key=True)   # 'SBER'
    source = Column(String(16), nullable=False)     # 'moex_iss'
    open = Column(Numeric(18, 6), nullable=True)
    high = Column(Numeric(18, 6), nullable=True)
    low = Column(Numeric(18, 6), nullable=True)
    close = Column(Numeric(18, 6), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<EquityPrice(ticker='{self.ticker}', date='{self.date}', close={self.close})>"


class CbrKeyRate(Base):
    """
    Ключевая ставка ЦБ РФ по датам ДЕЙСТВИЯ (не объявления решения) —
    источник: SOAP-сервис CBR DailyInfo, см. db/sync_reference_data.py.
    Разница дат объявления и действия — до нескольких дней, использование
    даты объявления было бы утечкой из будущего.
    """

    __tablename__ = "cbr_key_rate"

    date = Column(Date, primary_key=True)
    rate = Column(Numeric(6, 3), nullable=False)  # процент годовых
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<CbrKeyRate(date='{self.date}', rate={self.rate})>"


class ForeignKeyRate(Base):
    """
    Ставка иностранного ЦБ/денежного рынка. `series` — часть первичного
    ключа (не просто метаданные): для одной страны может быть несколько
    несовпадающих рядов (например, LPR и SHIBOR для Китая) — это разные
    показатели, не дубли одного и того же, поэтому оба должны иметь право
    сосуществовать в таблице одновременно. См. db/sync_reference_data.py.
    """

    __tablename__ = "foreign_key_rates"

    date = Column(Date, primary_key=True)
    country = Column(String(8), primary_key=True)    # 'CN'
    series = Column(String(32), primary_key=True)     # 'akshare_shibor_1y', 'akshare_lpr_1y'
    rate = Column(Numeric(6, 3), nullable=False)       # процент годовых
    source = Column(String(16), nullable=False)        # 'akshare'
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ForeignKeyRate(country='{self.country}', series='{self.series}', date='{self.date}', rate={self.rate})>"


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
