import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from deprecation import DeprecatedWarning
from t_tech.invest import Client, InstrumentStatus
from t_tech.invest.schemas import Future
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.database import engine, Base

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecatedWarning)

load_dotenv()
TINKOFF_TOKEN = os.getenv("TINKOFF_SANDBOX_TOKEN")

# Синтетические "бессрочные" инструменты (exp=2099-12-31), которые API отдаёт
# вперемешку с реальными контрактами — не настоящие контракты, отбрасываем.
SYNTHETIC_EXPIRATION_YEAR_CUTOFF = 2090

# Root-серии, которые отслеживаем. Один словарь = одна строка в futures_roots
# плюс правило, как выбрать её контракты из общего списка API.
# Чтобы добавить новый инструмент позже (например GLDRUBF) — дописать сюда
# запись и перезапустить sync_all_roots(), ничего больше менять не нужно.
ROOT_SERIES = [
    {
        "root_symbol": "SBERF",
        "basic_asset": "SBER",              # точное совпадение — исключает 'SBERP'
        "ticker_prefix": None,
        "underlying_type": "share",
        "underlying_name": "Сбербанк (обыкновенные акции)",
        "description": "Этап 4 — фьючерс на акцию, сравнение с моделью акций из ../agent",
    },
    {
        "root_symbol": "CNYRUBF",
        "basic_asset": "CNY/RUB",           # точное совпадение — исключает 'USD/CNY'
        "ticker_prefix": None,
        "underlying_type": "currency",
        "underlying_name": "Юань/рубль",
        "description": "Этап 1 — валютный риск в чистом виде: курс + разница ставок ЦБ",
    },
    {
        "root_symbol": "IMOEXF",
        "basic_asset": "IMOEX",             # точное совпадение — исключает 'MOEX' (акция биржи)
        "ticker_prefix": None,
        "underlying_type": "index",
        "underlying_name": "Индекс МосБиржи",
        "description": "Этап 3 — риск на рынок целиком, нет объекта для дельта-хеджа отдельной бумагой",
    },
    {
        "root_symbol": "BR",
        "basic_asset": "Brent",
        "ticker_prefix": "BR",              # исключает 'BM' — мини-контракт без истории до 2025
        "underlying_type": "commodity",
        "underlying_name": "Нефть Brent",
        "description": "Этап 2 — контанго/бэквордация от стоимости хранения, влияние мировых цен",
    },
]


def _quotation_to_float(q) -> Optional[float]:
    if q is None:
        return None
    return q.units + q.nano / 1e9


def _match_contracts(all_futures: List[Future], rule: dict) -> List[Future]:
    matched = []
    for f in all_futures:
        if f.basic_asset != rule["basic_asset"]:
            continue
        if rule["ticker_prefix"] and not f.ticker.startswith(rule["ticker_prefix"]):
            continue
        if f.expiration_date.year >= SYNTHETIC_EXPIRATION_YEAR_CUTOFF:
            continue
        matched.append(f)
    return matched


def sync_all_roots(token: str = TINKOFF_TOKEN, root_series: List[dict] = ROOT_SERIES):
    """
    Тянет из T-Invest API все фьючерсы (включая истёкшие — INSTRUMENT_STATUS_ALL),
    фильтрует по правилам ROOT_SERIES и записывает через UPSERT в futures_roots
    и futures_contracts. Безопасно перезапускать: существующие строки обновятся,
    новые добавятся, дубликатов не будет.
    """
    print("📦 0. Создание базовых таблиц (если их ещё нет)...")
    from db.models import FuturesRoot, FuturesContract
    Base.metadata.create_all(bind=engine, tables=[FuturesRoot.__table__, FuturesContract.__table__])

    print("🔄 1. Загрузка полного списка фьючерсов из T-Invest (INSTRUMENT_STATUS_ALL)...")
    with Client(token) as client:
        response = client.instruments.futures(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_ALL)
        all_futures = response.instruments
    print(f"    Всего фьючерсов от API: {len(all_futures)}")

    total_contracts = 0
    with engine.begin() as conn:
        for rule in root_series:
            matched = _match_contracts(all_futures, rule)
            print(f"  === {rule['root_symbol']} ({rule['basic_asset']}): найдено {len(matched)} контрактов ===")

            root_stmt = pg_insert(FuturesRoot.__table__).values(
                root_symbol=rule["root_symbol"],
                basic_asset=rule["basic_asset"],
                underlying_type=rule["underlying_type"],
                underlying_name=rule["underlying_name"],
                description=rule["description"],
            )
            root_stmt = root_stmt.on_conflict_do_update(
                index_elements=["root_symbol"],
                set_={
                    "basic_asset": root_stmt.excluded.basic_asset,
                    "underlying_type": root_stmt.excluded.underlying_type,
                    "underlying_name": root_stmt.excluded.underlying_name,
                    "description": root_stmt.excluded.description,
                },
            )
            conn.execute(root_stmt)

            for f in matched:
                contract_stmt = pg_insert(FuturesContract.__table__).values(
                    figi=f.figi,
                    root_symbol=rule["root_symbol"],
                    ticker=f.ticker,
                    name=f.name,
                    expiration_date=f.expiration_date.date(),
                    first_trade_date=f.first_trade_date.date() if f.first_trade_date and f.first_trade_date.year > 1971 else None,
                    last_trade_date=f.last_trade_date.date() if f.last_trade_date and f.last_trade_date.year > 1971 else None,
                    currency=f.currency,
                    lot=f.lot,
                    min_price_increment=_quotation_to_float(f.min_price_increment),
                    api_trade_available_flag=f.api_trade_available_flag,
                )
                contract_stmt = contract_stmt.on_conflict_do_update(
                    index_elements=["figi"],
                    set_={
                        "ticker": contract_stmt.excluded.ticker,
                        "name": contract_stmt.excluded.name,
                        "expiration_date": contract_stmt.excluded.expiration_date,
                        "first_trade_date": contract_stmt.excluded.first_trade_date,
                        "last_trade_date": contract_stmt.excluded.last_trade_date,
                        "currency": contract_stmt.excluded.currency,
                        "lot": contract_stmt.excluded.lot,
                        "min_price_increment": contract_stmt.excluded.min_price_increment,
                        "api_trade_available_flag": contract_stmt.excluded.api_trade_available_flag,
                    },
                )
                conn.execute(contract_stmt)
                total_contracts += 1

    print(f"🎉 Готово: {len(root_series)} root-серий, {total_contracts} контрактов записано/обновлено.")


if __name__ == "__main__":
    sync_all_roots()
