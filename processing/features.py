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
    ROOT_TO_PRICE_SCALE,
    ROOT_TO_SPOT_EQUITY,
    ROOT_TO_SPOT_INDEX,
    ROOT_TO_SPOT_PAIR,
    get_equity_map,
    get_index_map,
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

        # масштаб котировки к единицам спота (у MIX — пункты индекса × 100)
        actual = b["close"] / ROOT_TO_PRICE_SCALE.get(root_symbol, 1.0)
        theory_rub_only = spot * (1 + (r_rub / 100) * t_days / 365)
        b["basis_pct_rub_only"] = (actual - theory_rub_only) / theory_rub_only * 100

        # Без второй ставки полная формула вырождается в рублёвую: у фьючерса
        # на ИНДЕКС нет "иностранной ставки", вместо неё из справедливой цены
        # надо было бы вычесть ожидаемые дивиденды — их в проекте нет (см.
        # раздел 26 журнала, это известное ограничение). Чтобы модели Этапа 1
        # (они читают basis_pct_full) работали и здесь, дублируем значение.
        b["basis_pct_full"] = b["basis_pct_rub_only"] if not fr_dates else None
        if fr_dates:
            r_cn = rate_as_of(d, fr_dates, fr_rates)
            if r_cn is not None:
                theory_full = theory_rub_only / (1 + (r_cn / 100) * t_days / 365)
                b["basis_pct_full"] = (actual - theory_full) / theory_full * 100


def load_basis_inputs(root_symbol: str):
    """None, если для root-серии нет спот-пары (базис не считается)."""
    spot_pair = ROOT_TO_SPOT_PAIR.get(root_symbol)
    index_symbol = ROOT_TO_SPOT_INDEX.get(root_symbol)
    equity_ticker = ROOT_TO_SPOT_EQUITY.get(root_symbol)
    if spot_pair is None and index_symbol is None and equity_ticker is None:
        return None
    if spot_pair:
        spot_map = get_spot_map(spot_pair)
    elif index_symbol:
        spot_map = get_index_map(index_symbol)
    else:
        spot_map = get_equity_map(equity_ticker)
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
    "carry_annual",
    "log_return_1", "volatility_20", "momentum_10",
]


# ------------------------------------------------- производные (стационарные)
#
# Признаки выше — УРОВНИ (сырой базис, сырая волатильность, сырые дни до
# экспирации). Проверка adversarial validation (plan.md, раздел 20) показала:
# распределения уровней между train и test различимы с AUC 0.96 — модель,
# обученная на одном режиме, применяется к другому и экстраполирует вслепую.
#
# Ниже — те же величины, переведённые в ОТНОСИТЕЛЬНЫЕ единицы: "насколько
# текущее значение необычно относительно недавнего прошлого", а не "чему оно
# равно". Все окна причинные (rolling по прошлым барам, current бар включён,
# будущее не используется) и считаются ВНУТРИ КОНТРАКТА — окно не
# перепрыгивает через ролл, как и у ценовых признаков выше.

DERIVED_FEATURES = [
    "basis_z",        # z-score базиса по скользящему окну своего контракта
    "basis_chg_20",   # изменение базиса за 20 баров (разность — стационарна)
    "vol_ratio",      # log(volatility_20 / её скользящая медиана) — режим вола
    "ret_1_z",        # log_return_1 в единицах текущей волатильности
    "mom_10_z",       # momentum_10 в единицах сигмы (нормировка на sqrt(10))
    "mom_50_z",       # то же на более длинном окне — раздел 14 (momentum_10
                      # почти не работал, возможно окно слишком короткое)
    "carry_z",        # z-score carry_annual — «базис по кривой», работает без
                      # внешнего спота, т.е. и там, где ROOT_TO_SPOT_PAIR пуст
    "carry_chg_20",   # изменение carry за 20 баров
]


def add_derived_features(df, window: int = 250, min_periods: int = 60):
    """Добавляет DERIVED_FEATURES в pandas-таблицу признаков (in place → df).

    df должен быть отсортирован по (ticker, time) и содержать колонки
    basis_pct_full, volatility_20, log_return_1, momentum_10.
    """
    import numpy as np

    df = df.sort_values(["ticker", "time"]).reset_index(drop=True)
    g = df.groupby("ticker", sort=False)

    roll = lambda col, fn: g[col].transform(
        lambda s: getattr(s.rolling(window, min_periods=min_periods), fn)()
    )

    basis_mean = roll("basis_pct_full", "mean")
    basis_std = roll("basis_pct_full", "std")
    # std ~ 0 (базис стоит на месте) дал бы бесконечный z — такие бары в NaN.
    basis_std = basis_std.where(basis_std > 1e-9)
    df["basis_z"] = ((df["basis_pct_full"] - basis_mean) / basis_std).clip(-10, 10)

    df["basis_chg_20"] = g["basis_pct_full"].transform(lambda s: s - s.shift(20))

    # carry_annual (календарный спред, scripts/build_continuous_features.py) —
    # у старых витрин этой колонки нет вообще, поэтому её отсутствие не ошибка,
    # а «признак не считался»: тогда производные от неё просто NaN.
    if "carry_annual" not in df.columns:
        df["carry_annual"] = float("nan")
    carry_mean = roll("carry_annual", "mean")
    carry_std = roll("carry_annual", "std").where(lambda s: s > 1e-9)
    df["carry_z"] = ((df["carry_annual"] - carry_mean) / carry_std).clip(-10, 10)
    df["carry_chg_20"] = g["carry_annual"].transform(lambda s: s - s.shift(20))

    vol_med = roll("volatility_20", "median")
    vol_med = vol_med.where(vol_med > 1e-12)
    df["vol_ratio"] = np.log(df["volatility_20"] / vol_med).clip(-5, 5)

    vol = df["volatility_20"].where(df["volatility_20"] > 1e-12)
    df["ret_1_z"] = (df["log_return_1"] / vol).clip(-10, 10)
    df["mom_10_z"] = (df["momentum_10"] / (vol * math.sqrt(10))).clip(-10, 10)

    mom_50 = g["log_return_1"].transform(lambda s: s.rolling(50, min_periods=50).sum())
    df["mom_50_z"] = (mom_50 / (vol * math.sqrt(50))).clip(-10, 10)

    return df
