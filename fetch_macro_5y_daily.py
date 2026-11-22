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

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
MERGED_MACRO_CSV = OUTPUT_DIR / "macro_daily_5y.csv"
OVERLAP_DAYS = 7
MAX_WORKERS = 6

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


def fetch_yahoo_close(symbol: str, period1: int | None = None, period2: int | None = None) -> pd.Series:
    """Fetch daily settled close prices for a Yahoo Finance symbol over a 5y or bounded window."""
    encoded = urllib.parse.quote(symbol, safe="")
    if period1 is not None and period2 is not None:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?period1={period1}&period2={period2}&interval=1d"
    else:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?range=5y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    closes = (result.get("indicators", {}).get("quote") or [{}])[0].get("close") or []

    records = {}
    for ts, c in zip(timestamps, closes):
        if c is None:
            continue
        d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        records[d] = round(float(c), 4)

    if not records:
        return pd.Series(dtype=float)

    s = pd.Series(records, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    today = pd.Timestamp.today().normalize()
    s = s[s.index < today]
    return s


def fetch_yahoo_close_5y(symbol: str) -> pd.Series:
    """Backwards-compatible wrapper for fetching full 5-year close history."""
    return fetch_yahoo_close(symbol)


def merge_macro_series(existing: pd.Series | None, incoming: pd.Series) -> pd.Series:
    """Merge existing and incoming macro series, letting incoming overwrite overlapping dates."""
    if existing is None or existing.empty:
        return incoming.sort_index()
    if incoming.empty:
        return existing.sort_index()
    return incoming.combine_first(existing).sort_index()


def _load_existing_macro_df() -> pd.DataFrame | None:
    if not MERGED_MACRO_CSV.exists() or MERGED_MACRO_CSV.stat().st_size == 0:
        return None
    df = pd.read_csv(MERGED_MACRO_CSV, parse_dates=["date"])
    if df.empty or "date" not in df.columns:
        return None
    return df.set_index("date").sort_index()


def refresh_macro_data(full: bool = False, verbose: bool = True) -> dict:
    """Concurrently refresh all 12 macro series via incremental delta or full 5y fetches."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_df = None if full else _load_existing_macro_df()
    period2 = int(time.time())

    fetch_tasks: dict[str, tuple[str, str, int | None, pd.Series | None]] = {}
    any_incremental = False
    for col, (symbol, label) in TICKERS.items():
        old_s: pd.Series | None = None
        p1: int | None = None
        if existing_df is not None and col in existing_df.columns:
            non_null = existing_df[col].dropna()
            if not non_null.empty:
                old_s = non_null.astype(float)
                last_dt = non_null.index.max() - pd.Timedelta(days=OVERLAP_DAYS)
                p1 = int(last_dt.tz_localize("UTC").timestamp()) if last_dt.tzinfo is None else int(last_dt.timestamp())
                any_incremental = True
        fetch_tasks[col] = (symbol, label, p1, old_s)

    mode = "incremental" if any_incremental else "full"
    series_map: dict[str, pd.Series] = {}
    failed_tickers: list[str] = []
    all_ok = True
    fetched_bars = 0

    if verbose:
        print(f"Fetching {mode} daily expanded macro predictors (12 free Yahoo Finance series):")

    def _worker(item: tuple[str, tuple[str, str, int | None, pd.Series | None]]):
        col, (symbol, label, p1, old_s) = item
        new_s = fetch_yahoo_close(symbol, period1=p1, period2=period2 if p1 is not None else None)
        return col, symbol, label, old_s, new_s

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, item): item for item in fetch_tasks.items()}
        for col, (symbol, label, _, old_s) in fetch_tasks.items():
            # Find matching future for deterministic log order
            fut = next(f for f, it in futures.items() if it[0] == col)
            try:
                _, _, _, _, new_s = fut.result()
                fetched_bars += len(new_s)
                merged_s = merge_macro_series(old_s, new_s)
                if not merged_s.empty:
                    series_map[col] = merged_s
                if verbose and not merged_s.empty:
                    print(
                        f"  [SUCCESS] {label:32s} ({symbol:8s}) | "
                        f"delta={len(new_s):4d} | total={len(merged_s):4d} | "
                        f"{merged_s.index[0].date()} -> {merged_s.index[-1].date()} | "
                        f"latest={merged_s.iloc[-1]:,.4f}"
                    )
            except Exception as err:
                all_ok = False
                failed_tickers.append(col)
                if old_s is not None and not old_s.empty:
                    series_map[col] = old_s
                if verbose:
                    print(f"  [FAILED ] {label:32s} ({symbol:8s}) | {err}")

    if not series_map:
        if verbose:
            print("STATUS: FAILED (No macro series retrieved)")
        return {
            "all_ok": False,
            "mode": mode,
            "total_rows": 0,
            "cols": 0,
            "fetched_bars": fetched_bars,
            "failed_tickers": failed_tickers,
        }

    macro_df = pd.DataFrame(series_map).sort_index()
    macro_df.index.name = "date"
    macro_out = macro_df.reset_index()
    macro_out["date"] = macro_out["date"].dt.strftime("%Y-%m-%d")
    macro_out.to_csv(MERGED_MACRO_CSV, index=False)

    if verbose:
        print(f"\nSTATUS: {'SUCCESS' if all_ok else 'PARTIAL'} ({mode})")
        print(f"Saved expanded macro CSV: {MERGED_MACRO_CSV} ({len(macro_out):,} rows x {macro_out.shape[1]} cols)")

    return {
        "all_ok": all_ok,
        "mode": mode,
        "total_rows": len(macro_out),
        "cols": int(macro_out.shape[1]),
        "fetched_bars": fetched_bars,
        "failed_tickers": failed_tickers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch 12 macro & cross-asset series from Yahoo Finance.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force a full 5-year refetch instead of incremental delta updates.",
    )
    args = parser.parse_args(argv)
    summary = refresh_macro_data(full=args.full, verbose=True)
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
