"""
Fetch 5-year daily data for 12 key Gold macro & cross-asset predictors from Yahoo Finance
(free, no API key required, repeatable anytime):
  1. dxy    : DX-Y.NYB  (US Dollar Index)
  2. us10y  : ^TNX      (10-Year US Treasury Yield)
  3. us5y   : ^FVX      (5-Year US Treasury Yield)
  4. us30y  : ^TYX      (30-Year US Treasury Yield)
  5. silver : SI=F      (Silver Futures)
  6. oil    : CL=F      (WTI Crude Oil Futures)
  7. copper : HG=F      (Copper Futures)
  8. vix    : ^VIX      (CBOE Volatility Index / Fear Gauge)
  9. sp500  : ^GSPC     (S&P 500 Equity Index)
  10. gdx   : GDX       (VanEck Gold Miners ETF)
  11. eurusd: EURUSD=X  (EUR/USD Spot Exchange Rate)
  12. usdjpy: JPY=X     (USD/JPY Spot Exchange Rate)
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
MERGED_MACRO_CSV = OUTPUT_DIR / "macro_daily_5y.csv"

TICKERS = {
    "dxy": ("DX-Y.NYB", "US Dollar Index (DXY)"),
    "us10y": ("^TNX", "10-Year US Treasury Yield (%)"),
    "us5y": ("^FVX", "5-Year US Treasury Yield (%)"),
    "us30y": ("^TYX", "30-Year US Treasury Yield (%)"),
    "silver": ("SI=F", "Silver Futures (USD/oz)"),
    "oil": ("CL=F", "WTI Crude Oil Futures (USD/bbl)"),
    "copper": ("HG=F", "Copper Futures (USD/lb)"),
    "vix": ("^VIX", "CBOE Volatility Index (VIX)"),
    "sp500": ("^GSPC", "S&P 500 Index"),
    "gdx": ("GDX", "VanEck Gold Miners ETF"),
    "eurusd": ("EURUSD=X", "EUR/USD Exchange Rate"),
    "usdjpy": ("JPY=X", "USD/JPY Exchange Rate"),
}


def fetch_yahoo_close_5y(symbol: str) -> pd.Series:
    encoded = urllib.parse.quote(symbol, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?range=5y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]

    records = {}
    for ts, c in zip(timestamps, closes):
        if c is None:
            continue
        d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        records[d] = round(float(c), 4)

    s = pd.Series(records, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    today = pd.Timestamp.today().normalize()
    s = s[s.index < today]
    return s


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    series_map: dict[str, pd.Series] = {}
    all_ok = True

    print("Fetching 5-year daily expanded macro predictors (12 free Yahoo Finance series):")
    for col, (symbol, label) in TICKERS.items():
        try:
            s = fetch_yahoo_close_5y(symbol)
            series_map[col] = s
            print(
                f"  [SUCCESS] {label:32s} ({symbol:8s}) | "
                f"{len(s):4d} rows | {s.index[0].date()} -> {s.index[-1].date()} | "
                f"latest={s.iloc[-1]:,.4f}"
            )
        except Exception as err:
            all_ok = False
            print(f"  [FAILED ] {label:32s} ({symbol:8s}) | {err}")

    if not series_map:
        print("STATUS: FAILED (No macro series retrieved)")
        return 1

    macro_df = pd.DataFrame(series_map).sort_index()
    macro_df.index.name = "date"
    macro_out = macro_df.reset_index()
    macro_out["date"] = macro_out["date"].dt.strftime("%Y-%m-%d")
    macro_out.to_csv(MERGED_MACRO_CSV, index=False)

    print(f"\nSTATUS: {'SUCCESS' if all_ok else 'PARTIAL'}")
    print(f"Saved expanded macro CSV: {MERGED_MACRO_CSV} ({len(macro_out):,} rows x {macro_out.shape[1]} cols)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
