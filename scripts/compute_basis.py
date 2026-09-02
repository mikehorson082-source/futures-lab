"""
Этап 1 (CNYRUBF) — теоретический базис против фактического.

Теория (covered interest rate parity, Hull): справедливая цена фьючерса

    F_theory_rub_only = S * (1 + r_RUB * T/365)                     — упрощённая версия
    F_theory_full     = S * (1 + r_RUB * T/365) / (1 + r_CN * T/365) — полная формула

где S — спот-курс, T — календарных дней до экспирации, r_RUB/r_CN —
безрисковые ставки (в долях). Считаются ОБЕ версии одновременно, чтобы
можно было прямо сравнить: помогает ли ставка Китая объяснить дрейф знака
базиса, обнаруженный на первом проходе (см. plan.md, раздел 9.5), или нет.

    basis_pct = (F_actual - F_theory) / F_theory * 100

r_CN — приближение (FRED `INTDSRCNM193N`, месячная гранулярность, ряд
обрывается на 2025-06 — после этой даты используется forward-fill
последнего известного значения). Подробности и почему не SHIBOR/API
Ninjas — db/sync_reference_data.py и plan.md, раздел 9.

Только чтение из БД, ничего не пишет. В отличие от scripts/diagnose_roll_splice.py
здесь НЕТ заранее решённых порогов "хорошо/плохо" — числа печатаются как
есть, без вердикта с придуманным порогом.

Запуск:
    .venv/bin/python -m scripts.compute_basis --root CNYRUBF
"""
import argparse
import bisect
import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from db.database import engine

# Какой спот-курс/иностранная ставка соответствует какой root-серии
# фьючерса. Единственная запись сейчас — CNYRUBF; добавлять сюда по мере
# расширения на другие валютные серии.
ROOT_TO_SPOT_PAIR = {
    "CNYRUBF": "CNYRUB",
}
# Серии, у которых "спот" — биржевой ИНДЕКС (таблица index_prices), а не
# валютная пара. Формула базиса та же, источник другой.
ROOT_TO_SPOT_INDEX = {
    "IMOEXF": "IMOEX",
}
# Масштаб котировки фьючерса к единицам спота. У MIX (фьючерс на индекс
# МосБиржи) цена выражена в пунктах индекса × 100: MXU6 = 209896 при индексе
# 2093.4. Без деления на 100 базис был бы бессмысленным (+9800%).
ROOT_TO_PRICE_SCALE = {
    "IMOEXF": 100.0,
    "SBERF": 100.0,   # SRU6 = 27270 при цене акции 270.11 -> контракт на 100 акций
}
# Серии, у которых "спот" — цена АКЦИИ (таблица equity_prices).
ROOT_TO_SPOT_EQUITY = {
    "SBERF": "SBER",
}
ROOT_TO_FOREIGN_RATE = {
    # (country, series) — series см. db/sync_reference_data.py. SHIBOR 1Y —
    # дневная гранулярность и нет обрыва истории (в отличие от прежней
    # попытки через FRED и от LPR, который обновляется раз в месяц).
    "CNYRUBF": ("CN", "akshare_shibor_1y"),
}

# Бакеты "дней до экспирации" для сравнения поведения базиса вблизи/вдали
# от неё — просто разбивка для отображения, не порог принятия решения.
DAYS_TO_EXPIRY_BUCKETS = [7, 30, 90, 180]


def get_contracts(root_symbol: str):
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT figi, ticker, expiration_date
            FROM futures_contracts
            WHERE root_symbol = :root
            ORDER BY expiration_date
        """), {"root": root_symbol}).fetchall()


def get_daily_closes(root_symbol: str):
    """Последняя свеча каждого торгового дня для каждого контракта серии —
    один проход по futures_candles вместо запроса на контракт."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (fc.figi, fcd.time::date)
                fc.figi, fcd.time::date AS date, fcd.close
            FROM futures_candles fcd
            JOIN futures_contracts fc USING (figi)
            WHERE fc.root_symbol = :root
            ORDER BY fc.figi, fcd.time::date, fcd.time DESC
        """), {"root": root_symbol}).fetchall()
    by_figi = {}
    for r in rows:
        by_figi.setdefault(r.figi, []).append((r.date, float(r.close)))
    return by_figi


def get_spot_map(pair: str):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, close FROM currency_rates WHERE pair = :pair
        """), {"pair": pair}).fetchall()
    return {r.date: float(r.close) for r in rows}


def get_index_map(symbol: str):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, close FROM index_prices WHERE symbol = :symbol
        """), {"symbol": symbol}).fetchall()
    return {r.date: float(r.close) for r in rows}


def get_equity_map(ticker: str):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, close FROM equity_prices WHERE ticker = :ticker
        """), {"ticker": ticker}).fetchall()
    return {r.date: float(r.close) for r in rows}


def get_rate_series(table: str, extra_where: str = "", params: dict | None = None):
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, rate FROM {table} {extra_where} ORDER BY date
        """), params or {}).fetchall()
    dates = [r.date for r in rows]
    rates = [float(r.rate) for r in rows]
    return dates, rates


def rate_as_of(d, dates, rates):
    """Последняя известная на дату d ставка (dates отсортированы по возрастанию)
    — реализует forward-fill автоматически: для дат позже последней точки
    возвращается последнее известное значение."""
    i = bisect.bisect_right(dates, d) - 1
    if i < 0:
        return None
    return rates[i]


def compute_basis(root_symbol: str):
    contracts = get_contracts(root_symbol)
    spot_pair = ROOT_TO_SPOT_PAIR.get(root_symbol)
    if spot_pair is None:
        raise ValueError(f"Нет соответствия спот-парой для {root_symbol} — добавь в ROOT_TO_SPOT_PAIR")
    foreign_rate_key = ROOT_TO_FOREIGN_RATE.get(root_symbol)

    daily_closes = get_daily_closes(root_symbol)
    spot_map = get_spot_map(spot_pair)
    kr_dates, kr_rates = get_rate_series("cbr_key_rate")
    foreign_country = None
    fr_dates, fr_rates = ([], [])
    if foreign_rate_key:
        foreign_country, foreign_series = foreign_rate_key
        fr_dates, fr_rates = get_rate_series(
            "foreign_key_rates", "WHERE country = :country AND series = :series",
            {"country": foreign_country, "series": foreign_series},
        )

    per_contract = []
    skipped_no_spot = 0
    skipped_no_rate = 0
    skipped_expired_before_data = 0

    for c in contracts:
        closes = daily_closes.get(c.figi, [])
        rows = []
        for d, actual in closes:
            t_days = (c.expiration_date - d).days
            if t_days < 0:
                skipped_expired_before_data += 1
                continue
            spot = spot_map.get(d)
            if spot is None:
                skipped_no_spot += 1
                continue
            r_rub = rate_as_of(d, kr_dates, kr_rates)
            if r_rub is None:
                skipped_no_rate += 1
                continue

            theory_rub_only = spot * (1 + (r_rub / 100) * t_days / 365)
            basis_pct_rub_only = (actual - theory_rub_only) / theory_rub_only * 100

            basis_pct_full = None
            if foreign_country:
                r_cn = rate_as_of(d, fr_dates, fr_rates)
                if r_cn is not None:
                    theory_full = theory_rub_only / (1 + (r_cn / 100) * t_days / 365)
                    basis_pct_full = (actual - theory_full) / theory_full * 100

            rows.append({"date": d, "t_days": t_days, "actual": actual, "spot": spot,
                         "basis_pct_rub_only": basis_pct_rub_only, "basis_pct_full": basis_pct_full})
        if rows:
            per_contract.append({"ticker": c.ticker, "expiration_date": c.expiration_date, "rows": rows})

    return per_contract, {
        "skipped_no_spot": skipped_no_spot,
        "skipped_no_rate": skipped_no_rate,
        "skipped_expired_before_data": skipped_expired_before_data,
        "has_foreign_rate": bool(foreign_country and fr_dates),
    }


def _mean_std(vals):
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def print_per_contract(per_contract, has_foreign_rate):
    print(f"\n=== Базис по контрактам ===\n")
    for c in per_contract:
        vals_rub = [r["basis_pct_rub_only"] for r in c["rows"]]
        mean_rub, std_rub = _mean_std(vals_rub)
        line = (f"  {c['ticker']:<8} эксп. {c['expiration_date']}  {len(vals_rub):>4} дн.  "
                f"rub_only: mean {mean_rub:+6.3f}% std {std_rub:5.3f}%")
        if has_foreign_rate:
            vals_full = [r["basis_pct_full"] for r in c["rows"] if r["basis_pct_full"] is not None]
            if vals_full:
                mean_full, std_full = _mean_std(vals_full)
                line += f"   full(+CN): mean {mean_full:+6.3f}% std {std_full:5.3f}%"
        print(line)


def print_overall(per_contract, has_foreign_rate):
    all_rub = [r["basis_pct_rub_only"] for c in per_contract for r in c["rows"]]
    if not all_rub:
        print("\n  Нет данных для общей статистики.")
        return
    mean_rub, std_rub = _mean_std(all_rub)
    print(f"\n=== Итог по всем контрактам ({len(all_rub)} контракто-дней) ===")
    print(f"  rub_only:  mean {mean_rub:+.3f}%  std {std_rub:.3f}%  min {min(all_rub):+.3f}%  max {max(all_rub):+.3f}%")
    if has_foreign_rate:
        all_full = [r["basis_pct_full"] for c in per_contract for r in c["rows"] if r["basis_pct_full"] is not None]
        if all_full:
            mean_full, std_full = _mean_std(all_full)
            print(f"  full(+CN): mean {mean_full:+.3f}%  std {std_full:.3f}%  min {min(all_full):+.3f}%  max {max(all_full):+.3f}%")


def print_drift_by_year(per_contract, has_foreign_rate):
    """Средний базис по году экспирации контракта — прямая проверка гипотезы
    про дрейф знака (раздел 9.5 plan.md): смотрим, сжимается ли разброс
    между годами при переходе от rub_only к full(+CN)."""
    print(f"\n=== Дрейф по году экспирации контракта (проверка гипотезы) ===")
    by_year = {}
    for c in per_contract:
        y = c["expiration_date"].year
        by_year.setdefault(y, {"rub": [], "full": []})
        for r in c["rows"]:
            by_year[y]["rub"].append(r["basis_pct_rub_only"])
            if r["basis_pct_full"] is not None:
                by_year[y]["full"].append(r["basis_pct_full"])
    for y in sorted(by_year):
        rub_vals = by_year[y]["rub"]
        mean_rub = statistics.mean(rub_vals)
        line = f"  {y}:  rub_only mean {mean_rub:+6.3f}%"
        if has_foreign_rate and by_year[y]["full"]:
            mean_full = statistics.mean(by_year[y]["full"])
            line += f"   full(+CN) mean {mean_full:+6.3f}%   изменение {mean_full - mean_rub:+6.3f}п.п."
        print(line)


def print_by_days_to_expiry(per_contract, buckets):
    print(f"\n=== Базис (rub_only) по бакетам 'дней до экспирации' (все контракты вместе) ===")
    buckets = sorted(buckets)

    def bucket_label(t_days):
        for b in buckets:
            if t_days <= b:
                return f"<= {b}д" if b == buckets[0] else f"{buckets[buckets.index(b)-1]}-{b}д"
        return f"> {buckets[-1]}д"

    grouped = {}
    for c in per_contract:
        for r in c["rows"]:
            grouped.setdefault(bucket_label(r["t_days"]), []).append(r["basis_pct_rub_only"])

    order = [f"<= {buckets[0]}д"] + [f"{buckets[i-1]}-{buckets[i]}д" for i in range(1, len(buckets))] + [f"> {buckets[-1]}д"]
    for label in order:
        vals = grouped.get(label)
        if not vals:
            continue
        mean, std = _mean_std(vals)
        print(f"  {label:<10} {len(vals):>5} набл.  mean {mean:+6.3f}%  std {std:5.3f}%  "
              f"mean|basis| {statistics.mean([abs(v) for v in vals]):5.3f}%")


def main():
    parser = argparse.ArgumentParser(description="Теоретический vs фактический базис фьючерса")
    parser.add_argument("--root", default="CNYRUBF", help="root_symbol серии")
    parser.add_argument("--days-to-expiry-buckets", type=int, nargs="+", default=DAYS_TO_EXPIRY_BUCKETS)
    args = parser.parse_args()

    per_contract, skipped = compute_basis(args.root)
    has_foreign_rate = skipped["has_foreign_rate"]

    print_per_contract(per_contract, has_foreign_rate)
    print_overall(per_contract, has_foreign_rate)
    if has_foreign_rate:
        print_drift_by_year(per_contract, has_foreign_rate)
    print_by_days_to_expiry(per_contract, args.days_to_expiry_buckets)

    print(f"\n  (пропущено: {skipped['skipped_no_spot']} дней без спот-курса, "
          f"{skipped['skipped_no_rate']} без ставки ЦБ, "
          f"{skipped['skipped_expired_before_data']} после формальной экспирации)")


if __name__ == "__main__":
    main()
