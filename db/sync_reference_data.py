"""
Вспомогательные (не фьючерсные) данные: курсы валют, ставки ЦБ.

Общий файл для всего, что не является фьючерсным контрактом или свечой —
сюда же в будущем добавляются новые загрузчики того же рода (например,
ставка PBOC, если понадобится для Этапа 1, или другие валютные пары).

Источники — оба публичные, без токена:
* Спот-курс валютной пары — MOEX ISS (`iss.moex.com`), тот же класс
  эндпоинта, что `dailytable` в db/sync_calendar.py.
* Ключевая ставка ЦБ РФ — SOAP-сервис CBR `DailyInfoWebServ` (`cbr.ru`),
  ряд по датам ДЕЙСТВИЯ ставки, не объявления решения (решение объявляется
  на несколько дней раньше, чем начинает действовать — использование даты
  объявления было бы утечкой из будущего). Тот же источник и та же причина,
  что в ../agent/db/sync_macro.py — решение сознательно перенесено оттуда.
* Ставка Китая (для полной формулы паритета ставок, Этап 1) — библиотека
  `akshare` (по просьбе пользователя, 2026-09-01), два ряда одновременно:
  **SHIBOR 1Y** (`macro_china_shibor_all`, дневная, 2022-01-04…сегодня, ни
  одного пропуска — источник jin10.com) и **LPR 1Y** (`macro_china_lpr`,
  по датам объявления ~раз в месяц, до 2026-08-20 — источник eastmoney.com).
  Оба сохраняются как отдельные `series` под `country='CN'` — не дубли, а
  разные показатели (см. db/models.py). `compute_basis.py` по умолчанию
  использует SHIBOR 1Y — дневная гранулярность и нет обрыва истории, в
  отличие от предыдущей попытки через FRED (`INTDSRCNM193N`: месячная,
  обрывалась на 2025-06). До akshare пробовали API Ninjas (изначально
  предложенный пользователем источник) — на бесплатном плане отдаёт только
  ТЕКУЩУЮ ставку, история требует Premium.
  ⚠️ **Тяжёлая зависимость:** akshare тянет pandas, numpy, lxml,
  beautifulsoup4, curl-cffi и mini-racer (встроенный движок JS — нужен для
  обхода антибот-защиты некоторых китайских сайтов-агрегаторов данных).
  Обоснованно только тем, что явно попросил пользователь; для одного ряда
  ставки это тяжеловесный набор зависимостей.

Загрузка спот-курса/ставки ЦБ РФ — через curl в subprocess, не urllib
напрямую: в этом окружении прямой urllib к внешним хостам стабильно уходит
в таймаут (та же особенность, что у db/sync_calendar.py и
db/sync_candles_zip.py). Ставка Китая — через akshare, у него свой HTTP-слой.

Запуск:
    .venv/bin/python -m db.sync_reference_data
"""
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.database import engine, Base, get_db_session
from db.models import CurrencyRate, CbrKeyRate, ForeignKeyRate

DEFAULT_START_DATE = "2022-01-01"  # тот же период, что у остальных данных проекта


def _curl(cmd_extra: list[str], timeout_sec: int = 25, max_retries: int = 3) -> bytes:
    """Общий curl-раннер: -4 (форсировать IPv4) — обязательный флаг в этом
    окружении, см. докстринг файла. С ретраями — ISS периодически отдаёт
    пустой ответ без ошибки curl (как в db/sync_candles_zip.py)."""
    cmd = ["curl", "-4", "-sS", "-m", str(timeout_sec - 5)] + cmd_extra
    last_output = b""
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
            last_output = result.stdout
            if last_output:
                return last_output
        except subprocess.TimeoutExpired:
            pass
        print(f"    ⚠️ Пустой/неудачный ответ, повтор {attempt}/{max_retries}...", flush=True)
        time.sleep(3)
    return last_output


def fetch_currency_rate(pair: str, ticker: str, board: str, start: str, end: str,
                         engine_name: str = "currency", market: str = "selt") -> list[dict]:
    """
    Постраничная выгрузка дневной истории валютной пары из MOEX ISS.
    Пагинация через `start=` (номер строки), страница — 100 строк.
    """
    rows = []
    cursor = 0
    while True:
        url = (
            f"https://iss.moex.com/iss/history/engines/{engine_name}/markets/{market}"
            f"/boards/{board}/securities/{ticker}.json"
            f"?from={start}&till={end}&start={cursor}&iss.meta=off&iss.only=history"
        )
        raw = _curl([url])
        payload = json.loads(raw)
        data = payload["history"]["data"]
        if not data:
            break
        cols = payload["history"]["columns"]
        i_date, i_board = cols.index("TRADEDATE"), cols.index("BOARDID")
        i_open, i_low, i_high, i_close = (cols.index("OPEN"), cols.index("LOW"),
                                           cols.index("HIGH"), cols.index("CLOSE"))
        for r in data:
            if r[i_board] != board or r[i_close] is None:
                continue
            rows.append({
                "date": date.fromisoformat(r[i_date]),
                "pair": pair,
                "source": "moex_iss",
                "open": r[i_open],
                "low": r[i_low],
                "high": r[i_high],
                "close": r[i_close],
            })
        print(f"    … {ticker}: {cursor + len(data)} строк", flush=True)
        cursor += len(data)
        if len(data) < 100:
            break
        time.sleep(0.2)
    return rows


def fetch_cbr_key_rate(start: str, end: str) -> list[dict]:
    """
    Ключевая ставка ЦБ РФ по датам действия, SOAP DailyInfoWebServ.KeyRate.
    Не постраничный — сервис отдаёт весь диапазон одним ответом (проверено
    на полном периоде 2022-2026, 1184 строки за один запрос).
    """
    body = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        '<soap:Body><KeyRate xmlns="http://web.cbr.ru/">'
        f"<fromDate>{start}</fromDate><ToDate>{end}</ToDate>"
        "</KeyRate></soap:Body></soap:Envelope>"
    )
    raw = _curl([
        "-X", "POST", "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx",
        "-H", "Content-Type: text/xml; charset=utf-8",
        "-H", "SOAPAction: http://web.cbr.ru/KeyRate",
        "-d", body,
    ], timeout_sec=40)

    root = ET.fromstring(raw)
    rows = []
    for kr in root.iter():
        if kr.tag.split("}")[-1] != "KR":
            continue
        dt = rate = None
        for child in kr:
            tag = child.tag.split("}")[-1]
            if tag == "DT":
                dt = date.fromisoformat(child.text[:10])
            elif tag == "Rate":
                rate = float(child.text)
        if dt is not None and rate is not None:
            rows.append({"date": dt, "rate": rate})
    return rows


def fetch_akshare_shibor_1y(start: str, end: str) -> list[dict]:
    """SHIBOR 1Y, дневная история (jin10.com через akshare). Полное покрытие
    без пропусков на всём периоде проекта (проверено 2026-09-01)."""
    import akshare as ak  # тяжёлый импорт (pandas/numpy/...) — только когда реально нужен
    df = ak.macro_china_shibor_all()
    df = df[(df["日期"] >= start) & (df["日期"] <= end)]
    rows = []
    for _, row in df.iterrows():
        rate = row["1Y-定价"]
        if rate is None or (isinstance(rate, float) and rate != rate):  # NaN
            continue
        rows.append({"date": date.fromisoformat(row["日期"]), "rate": float(rate)})
    return rows


def fetch_akshare_lpr_1y(start: str, end: str) -> list[dict]:
    """LPR 1Y (Loan Prime Rate), по датам объявления решений (eastmoney.com
    через akshare) — реже обновляется, чем SHIBOR, но это официальная
    ставка-бенчмарк, а не рыночная межбанковская."""
    import akshare as ak
    df = ak.macro_china_lpr()
    rows = []
    for _, row in df.iterrows():
        d, rate = row["TRADE_DATE"], row["LPR1Y"]
        if d is None or d < date.fromisoformat(start) or d > date.fromisoformat(end):
            continue
        if rate is None or (isinstance(rate, float) and rate != rate):  # NaN — до реформы LPR 2019 года
            continue
        rows.append({"date": d, "rate": float(rate)})
    return rows


def _upsert(table, rows: list[dict], conflict_cols: list[str]):
    if not rows:
        return
    db = get_db_session()
    try:
        stmt = pg_insert(table).values(rows)
        update_cols = {c.name: c for c in stmt.excluded if c.name not in conflict_cols}
        stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
        db.execute(stmt)
        db.commit()
    finally:
        db.close()


def sync_currency_rate(pair: str = "CNYRUB", ticker: str = "CNYRUB_TOM", board: str = "CETS",
                        start: str = DEFAULT_START_DATE, end: Optional[str] = None):
    end = end or date.today().isoformat()
    print(f"📥 Загрузка спот-курса {pair} ({ticker}, {start} … {end})...", flush=True)
    rows = fetch_currency_rate(pair, ticker, board, start, end)
    if not rows:
        print(f"⚠️ Пустой ответ ISS для {ticker} — не загружено.")
        return
    _upsert(CurrencyRate.__table__, rows, ["date", "pair"])
    print(f"✅ {pair}: {len(rows)} дней, {rows[-1]['date']} … {rows[0]['date']}, "
          f"close {rows[-1]['close']} … {rows[0]['close']}")


def sync_cbr_key_rate(start: str = DEFAULT_START_DATE, end: Optional[str] = None):
    end = end or date.today().isoformat()
    print(f"📥 Загрузка ключевой ставки ЦБ РФ ({start} … {end})...", flush=True)
    rows = fetch_cbr_key_rate(start, end)
    if not rows:
        print("⚠️ Пустой ответ CBR DailyInfo — не загружено.")
        return
    _upsert(CbrKeyRate.__table__, rows, ["date"])
    rows_sorted = sorted(rows, key=lambda r: r["date"])
    print(f"✅ Ключевая ставка: {len(rows)} дней, {rows_sorted[0]['date']} … {rows_sorted[-1]['date']}, "
          f"{rows_sorted[0]['rate']}% … {rows_sorted[-1]['rate']}%")


def sync_foreign_key_rate(country: str = "CN", start: str = DEFAULT_START_DATE, end: Optional[str] = None):
    end = end or date.today().isoformat()
    for series, fetch_fn, label in [
        ("akshare_shibor_1y", fetch_akshare_shibor_1y, "SHIBOR 1Y"),
        ("akshare_lpr_1y", fetch_akshare_lpr_1y, "LPR 1Y"),
    ]:
        print(f"📥 Загрузка ставки {country}/{label} (akshare, {start} … {end})...", flush=True)
        raw_rows = fetch_fn(start, end)
        if not raw_rows:
            print(f"⚠️ Пустой ответ akshare для {label} — не загружено.")
            continue
        rows = [{"date": r["date"], "country": country, "series": series,
                 "rate": r["rate"], "source": "akshare"} for r in raw_rows]
        _upsert(ForeignKeyRate.__table__, rows, ["date", "country", "series"])
        rows_sorted = sorted(rows, key=lambda r: r["date"])
        print(f"✅ {country}/{label}: {len(rows)} точек, "
              f"{rows_sorted[0]['date']} … {rows_sorted[-1]['date']}, "
              f"{rows_sorted[0]['rate']}% … {rows_sorted[-1]['rate']}%")


def sync_all(start: str = DEFAULT_START_DATE, end: Optional[str] = None):
    Base.metadata.create_all(bind=engine, tables=[CurrencyRate.__table__, CbrKeyRate.__table__, ForeignKeyRate.__table__])
    sync_currency_rate(start=start, end=end)
    sync_cbr_key_rate(start=start, end=end)
    sync_foreign_key_rate(start=start, end=end)


if __name__ == "__main__":
    sync_all()
