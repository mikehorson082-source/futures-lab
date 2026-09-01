"""
Этап 1 — разметка методом Triple Barrier (тройной барьер, López de Prado,
"Advances in Financial Machine Learning" — см. plan.md, раздел 4).

Идея: от каждого бара ставим два барьера — тейк-профит (TP) и стоп-лосс
(SL) — и смотрим, какой сработает ПЕРВЫМ в течение горизонта N баров
вперёд (путезависимо — важен порядок касаний, не только итоговая точка).
Если ни один барьер не сработал за горизонт — исход "timeout".

Решения под нашу специфику:
* Масштаб барьеров — уже посчитанный `volatility_20` (processing/features.py),
  не отдельный ATR: тот же смысл, не плодим второй индикатор ради него же.
* Группировка — по контракту (ticker), без понятия "сегмент остановки":
  у нас и так пул отдельных контрактов (раздел 3 plan.md), окно не
  переходит через ролл на другой контракт по построению.
* Размечается КАЖДЫЙ бар, не события (например, по CUSUM-фильтру). Проще
  для первого прохода, но это означает, что окна соседних меток сильно
  перекрываются (почти не независимы) — известная и осознанная
  провизорность. Не исправляется здесь выборкой событий, а
  КОМПЕНСИРУЕТСЯ весами (`compute_uniqueness_weights`, ниже, тоже AFML,
  гл. 4) — каждой метке присваивается вес по средней "уникальности" её
  окна, вместо того чтобы 8 почти одинаковых копий одного события
  считались 8 независимыми голосами при обучении.

Оба барьера — в терминах ДОХОДНОСТИ от close бара-события (не абсолютных
уровней в рублях) — так признак сравним между сериями разных типов
фьючерсов (доля, не рубли), что заранее заложено как принцип в plan.md,
раздел 3 ("проценты, а не абсолютные величины").

При касании обоих барьеров в одном и том же баре порядок по OHLC
невосстановим — берём худший случай (стоп).
"""
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def scan_barriers(
    bars: List[dict],
    horizon_bars: int = 20,
    tp_vol_mult: float = 2.0,
    sl_vol_mult: float = 1.0,
    require_full_horizon: bool = True,
    side: int = 1,
) -> List[dict]:
    """
    bars — бары ОДНОГО контракта, отсортированные по времени, с полями
    close/high/low/volatility_20 (processing/features.py).

    Возвращает НОВЫЙ список: копии баров-событий с добавленными полями
    target/t1/exit_price/exit_reason/bars_held/ret_gross/event_idx/exit_idx.
    Бары без volatility_20 (прогрев окна на первых барах контракта) и,
    если require_full_horizon=True, бары у которых не хватает полного
    окна вперёд до конца истории контракта — пропускаются (не размечены).

    side=+1 — ЛОНГ: TP выше цены входа (+tp_vol_mult*vol, касается high),
    SL ниже (-sl_vol_mult*vol, касается low).
    side=-1 — ШОРТ, ЗЕРКАЛЬНО: TP ниже (-tp_vol_mult*vol, касается low),
    SL выше (+sl_vol_mult*vol, касается high), доходность со знаком минус.

    Зачем нужна отдельная зеркальная разметка, а не "перевёрнутая
    вероятность лонговой модели": барьеры асимметричны (TP вдвое дальше
    SL), поэтому исход "SL сработал первым" означает движение вдвое
    МЕНЬШЕГО размера, чем TP. Торговать его в шорт — ловить половинное
    движение при тех же издержках на сделку (plan.md, раздел 25).
    """
    assert side in (1, -1), "side: +1 лонг, -1 шорт"
    n = len(bars)
    out = []

    for i in range(n):
        vol = bars[i].get("volatility_20")
        if vol is None or vol <= 0:
            continue
        if require_full_horizon and i + horizon_bars >= n:
            continue

        last = min(i + horizon_bars, n - 1)
        entry_close = bars[i]["close"]
        tp_level = entry_close * (1 + side * vol * tp_vol_mult)
        sl_level = entry_close * (1 - side * vol * sl_vol_mult)

        exit_idx, exit_reason, exit_price = None, None, None
        for j in range(i + 1, last + 1):
            if side == 1:
                hit_sl = bars[j]["low"] <= sl_level
                hit_tp = bars[j]["high"] >= tp_level
            else:
                hit_sl = bars[j]["high"] >= sl_level
                hit_tp = bars[j]["low"] <= tp_level
            if hit_sl:
                exit_idx, exit_reason, exit_price = j, "sl", sl_level
                break
            if hit_tp:
                exit_idx, exit_reason, exit_price = j, "tp", tp_level
                break
        if exit_idx is None:
            exit_idx, exit_reason, exit_price = last, "timeout", bars[last]["close"]

        row = dict(bars[i])
        row["target"] = 1 if exit_reason == "tp" else 0
        row["t1"] = bars[exit_idx]["time"]
        row["exit_price"] = exit_price
        row["exit_reason"] = exit_reason
        row["bars_held"] = exit_idx - i
        row["ret_gross"] = side * (exit_price - entry_close) / entry_close
        row["event_idx"] = i
        row["exit_idx"] = exit_idx
        out.append(row)

    return out


def overlap_stats(labels_by_ticker: Dict[str, List[dict]]) -> dict:
    """
    Приёмка: средний нахлёст окон меток (сколько других меток того же
    контракта начинаются раньше конца данной) — только чтобы честно
    показать масштаб проблемы (см. докстринг модуля), не для того, чтобы
    её решить.
    """
    per_label = []
    for rows in labels_by_ticker.values():
        if not rows:
            continue
        order = sorted(rows, key=lambda r: r["event_idx"])
        starts = [r["event_idx"] for r in order]
        ends = [r["exit_idx"] for r in order]
        for k, end in enumerate(ends):
            # сколько последующих меток стартуют раньше, чем закрылась текущая
            count = 0
            for s in starts[k + 1:]:
                if s <= end:
                    count += 1
                else:
                    break
            per_label.append(count)
    if not per_label:
        return {"mean_overlap": 0.0, "median_overlap": 0.0}
    per_label.sort()
    n = len(per_label)
    return {
        "mean_overlap": sum(per_label) / n * 2,
        "median_overlap": per_label[n // 2] * 2,
    }


def compute_uniqueness_weights(labels: List[dict], n_bars: int) -> None:
    """
    Добавляет каждой метке поле `sample_weight` — среднюю "уникальность"
    её окна (AFML, гл. 4): 1 / concurrency(t), усреднённое по всем барам
    t в [event_idx, exit_idx] метки, где concurrency(t) — сколько меток
    ТОГО ЖЕ контракта активны (их окно включает t) в этот момент.

    labels — вывод scan_barriers ДЛЯ ОДНОГО КОНТРАКТА (нужны event_idx/
    exit_idx, которые есть только внутри одного контракта — позиции в
    его собственной последовательности баров). n_bars — длина полной
    последовательности баров контракта (не только размеченных), чтобы
    массив concurrency покрывал весь диапазон возможных индексов.

    Смысл: метка, которая почти всё время висит в одиночестве (мало
    соседей одновременно открыто), получает вес ~1. Метка, которая
    почти всегда перекрывается с 7-8 другими (наша типичная ситуация,
    см. overlap_stats), получает вес ~0.1-0.15 — она несёт куда меньше
    независимой информации, чем формально одна строка датасета.
    """
    if not labels:
        return

    diff = [0] * (n_bars + 1)
    for r in labels:
        diff[r["event_idx"]] += 1
        diff[r["exit_idx"] + 1] -= 1

    concurrency = [0] * n_bars
    running = 0
    for t in range(n_bars):
        running += diff[t]
        concurrency[t] = running

    for r in labels:
        i, j = r["event_idx"], r["exit_idx"]
        span = concurrency[i:j + 1]
        r["sample_weight"] = sum(1.0 / c for c in span if c > 0) / len(span)


LABEL_COLUMNS = [
    "target", "t1", "exit_price", "exit_reason", "bars_held", "ret_gross", "sample_weight",
]
