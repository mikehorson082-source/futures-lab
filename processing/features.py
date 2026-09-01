"""
Этап 1 (CNYRUBF, но структурные и ценовые признаки общие на любую
root-серию) — признаки поверх dollar bars (processing/bars.py). Порядок
разработки — структурные, потом базис, потом ценовые (по просьбе
пользователя, 2026-09-01); в итоговой таблице все три группы считаются
вместе, один бар = одна строка.

СТРУКТУРНЫЕ (plan.md, раздел 3 — замена raw contract-id):
    days_to_expiration      — календарных дней от даты бара до экспирации.
    life_fraction_remaining — days_to_expiration / полная жизнь контракта
                               (first_trade_date..expiration_date), клип
                               [0, 1]. Переносится между контрактами разной
                               длины жизни лучше, чем абсолютные дни.
    contract_rank            — 1 = ближайший к экспирации из ещё не истёкших
                               контрактов root-серии на дату бара, 2 —
                               следующий и т.д. "Живой" контракт — тот, чей
                               first_trade_date <= дата <= expiration_date.
    volume_share              — доля дневного оборота этого контракта в
                               суммарном дневном обороте ВСЕХ живых
                               контрактов root-серии в этот день —
                               эмпирическая замена "типичной кривой
                               концентрации" из diagnose_roll_splice.py, без
                               зашитых бакетов дней.

БАЗИС — только для root-серий с записью в ROOT_TO_SPOT_PAIR
(scripts/compute_basis.py; сейчас только CNYRUBF). Формула и источники
данных переиспользуются оттуда напрямую (не дублируются), но F_actual
берётся из CLOSE БАРА, а не из дневного закрытия — на разрешении бара,
не дня. См. scripts/compute_basis.py и plan.md, раздел 9.

ЦЕНОВЫЕ — внутри контракта, БЕЗ протяжки через ролл (pool-архитектура,
раздел 3):
    log_return_1   — лог-доходность бара к предыдущему бару ТОГО ЖЕ
                      контракта.
    volatility_20  — pstdev log_return_1 за последние 20 баров контракта.
    momentum_10    — сумма log_return_1 за последние 10 баров контракта.
Все три — None на первых барах контракта (окно не должно перепрыгивать
через ролл на другой контракт) и там, где в окне есть None.

Минимальный стартовый набор ценовых признаков, не индикаторный зоопарк —
расширяется по мере необходимости, не заранее.
"""
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db.database import engine
from scripts.compute_basis import (
    ROOT_TO_FOREIGN_RATE,
    ROOT_TO_SPOT_PAIR,
    get_rate_series,
    get_spot_map,
    rate_as_of,
)


def get_contracts_meta(root_symbol: str) -> List[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT figi, ticker, expiration_date, first_trade_date
                FROM futures_contracts WHERE root_symbol = :root
                ORDER BY expiration_date
                """
            ),
            {"root": root_symbol},
        ).fetchall()
    return [
        {"figi": r.figi, "ticker": r.ticker, "expiration_date": r.expiration_date, "first_trade_date": r.first_trade_date}
        for r in rows
    ]


# ---------------------------------------------------------------- структурные

def build_daily_volume(bars_by_ticker: Dict[str, List[dict]]) -> Dict[str, Dict]:
    """ticker -> {date: суммарный dollar_volume баров этого контракта в этот день}."""
    daily = defaultdict(lambda: defaultdict(float))
    for ticker, bars in bars_by_ticker.items():
        for b in bars:
            daily[ticker][b["time"].date()] += b["dollar_volume"]
    return daily


def build_rank_and_share(contracts_meta: List[dict], daily_volume: Dict[str, Dict]) -> Dict[Tuple[str, object], Tuple[int, Optional[float]]]:
    """(ticker, date) -> (contract_rank, volume_share) по всем датам, где хоть один контракт торговал."""
    all_dates = set()
    for by_date in daily_volume.values():
        all_dates.update(by_date.keys())

    result = {}
    for d in all_dates:
        live = [
            c for c in contracts_meta
            if c["expiration_date"] >= d and (c["first_trade_date"] is None or c["first_trade_date"] <= d)
        ]
        live.sort(key=lambda c: c["expiration_date"])
        total_vol = sum(daily_volume.get(c["ticker"], {}).get(d, 0.0) for c in live)
        for rank, c in enumerate(live, start=1):
            vol = daily_volume.get(c["ticker"], {}).get(d, 0.0)
            share = vol / total_vol if total_vol > 0 else None
            result[(c["ticker"], d)] = (rank, share)
    return result


def add_structural_features(bars: List[dict], ticker: str, meta: dict, rank_share: dict) -> None:
    exp = meta["expiration_date"]
    first = meta["first_trade_date"] or exp
    total_life_days = max((exp - first).days, 1)
    for b in bars:
        d = b["time"].date()
        days_to_exp = (exp - d).days
        b["days_to_expiration"] = days_to_exp
        b["life_fraction_remaining"] = max(0.0, min(1.0, days_to_exp / total_life_days))
        rank, share = rank_share.get((ticker, d), (None, None))
        b["contract_rank"] = rank
        b["volume_share"] = share


# --------------------------------------------------------------------- базис

def add_basis_features(bars: List[dict], meta: dict, root_symbol: str, spot_map, kr_dates, kr_rates, fr_dates, fr_rates) -> None:
    exp = meta["expiration_date"]
    for b in bars:
        d = b["time"].date()
        t_days = (exp - d).days
        spot = spot_map.get(d)
        r_rub = rate_as_of(d, kr_dates, kr_rates)
        if spot is None or r_rub is None or t_days < 0:
            b["basis_pct_rub_only"] = None
            b["basis_pct_full"] = None
            continue

        actual = b["close"]
        theory_rub_only = spot * (1 + (r_rub / 100) * t_days / 365)
        b["basis_pct_rub_only"] = (actual - theory_rub_only) / theory_rub_only * 100

        b["basis_pct_full"] = None
        if fr_dates:
            r_cn = rate_as_of(d, fr_dates, fr_rates)
            if r_cn is not None:
                theory_full = theory_rub_only / (1 + (r_cn / 100) * t_days / 365)
                b["basis_pct_full"] = (actual - theory_full) / theory_full * 100


def load_basis_inputs(root_symbol: str):
    """None, если для root-серии нет спот-пары (базис не считается)."""
    spot_pair = ROOT_TO_SPOT_PAIR.get(root_symbol)
    if spot_pair is None:
        return None
    spot_map = get_spot_map(spot_pair)
    kr_dates, kr_rates = get_rate_series("cbr_key_rate")
    fr_dates, fr_rates = [], []
    foreign_key = ROOT_TO_FOREIGN_RATE.get(root_symbol)
    if foreign_key:
        country, series = foreign_key
        fr_dates, fr_rates = get_rate_series(
            "foreign_key_rates", "WHERE country = :country AND series = :series",
            {"country": country, "series": series},
        )
    return spot_map, kr_dates, kr_rates, fr_dates, fr_rates


# ------------------------------------------------------------------ ценовые

def add_price_features(bars: List[dict], vol_window: int = 20, mom_window: int = 10) -> None:
    closes = [b["close"] for b in bars]
    for i, b in enumerate(bars):
        b["log_return_1"] = None if i == 0 else math.log(closes[i] / closes[i - 1])

    for i, b in enumerate(bars):
        b["volatility_20"] = None
        if i >= vol_window:
            window = [bars[j]["log_return_1"] for j in range(i - vol_window + 1, i + 1)]
            if all(r is not None for r in window):
                b["volatility_20"] = statistics.pstdev(window)

    for i, b in enumerate(bars):
        b["momentum_10"] = None
        if i >= mom_window:
            window = [bars[j]["log_return_1"] for j in range(i - mom_window + 1, i + 1)]
            if all(r is not None for r in window):
                b["momentum_10"] = sum(window)


FEATURE_COLUMNS = [
    "time", "open_time", "ticker", "root_symbol",
    "open", "high", "low", "close", "volume", "dollar_volume", "n_ticks",
    "days_to_expiration", "life_fraction_remaining", "contract_rank", "volume_share",
    "basis_pct_rub_only", "basis_pct_full",
    "log_return_1", "volatility_20", "momentum_10",
]
