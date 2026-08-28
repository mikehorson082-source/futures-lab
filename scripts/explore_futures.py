import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from deprecation import DeprecatedWarning
from t_tech.invest import Client, InstrumentStatus

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecatedWarning)

load_dotenv()
TOKEN = os.getenv("TINKOFF_SANDBOX_TOKEN")

# Ключевые слова для базового актива каждой root-серии, которую обсуждали
TARGETS = {
    "SBER": "акция Сбербанк (этап 4)",
    "GAZP": "акция Газпром (этап 4, запасной вариант)",
    "CNY": "юань/рубль (этап 1)",
    "USD": "доллар/рубль (запасной вариант для этапа 1)",
    "MOEX": "индекс МосБиржи (этап 3)",
    "BR": "нефть Brent (этап 2)",
}

def main():
    with Client(TOKEN) as client:
        response = client.instruments.futures(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )
        futures = response.instruments

        print(f"Всего фьючерсов от API: {len(futures)}\n")

        by_basic_asset = {}
        for f in futures:
            by_basic_asset.setdefault(f.basic_asset, []).append(f)

        for keyword, label in TARGETS.items():
            print(f"=== {keyword} — {label} ===")
            matched_assets = sorted(a for a in by_basic_asset if keyword.upper() in a.upper())
            if not matched_assets:
                print("  (совпадений по basic_asset не найдено)")
                print()
                continue
            for asset in matched_assets:
                contracts = sorted(by_basic_asset[asset], key=lambda x: x.expiration_date)
                print(f"  basic_asset='{asset}': {len(contracts)} контрактов")
                for c in contracts:
                    exp = c.expiration_date.date() if c.expiration_date else "?"
                    print(f"    ticker={c.ticker:<16} figi={c.figi:<14} exp={exp} "
                          f"currency={c.currency} api_trade={c.api_trade_available_flag}")
            print()

if __name__ == "__main__":
    main()
