"""
Fetch the last 5 years of daily Gold Futures (GC=F) OHLCV data
from Yahoo Finance and save to data/gold_daily_5y.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=5y&interval=1d"
YAHOO_RANGE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?period1={period1}&period2={period2}&interval=1d"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = OUTPUT_DIR / "gold_daily_5y.csv"
OVERLAP_DAYS = 7
FIELDNAMES = ["date", "open", "high", "low", "close", "volume", "source"]


def load_existing_rows() -> list[dict]:
    """Load existing settled and raw GC=F rows from CSV_PATH if present."""
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return []

    rows: list[dict] = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            close_raw = raw.get("close", "")
            date_str = (raw.get("date") or "").strip()
            if not date_str or close_raw == "":
                continue
            rows.append(
                {
                    "date": date_str,
                    "open": round(float(raw["open"]), 4) if raw.get("open", "") != "" else "",
                    "high": round(float(raw["high"]), 4) if raw.get("high", "") != "" else "",
                    "low": round(float(raw["low"]), 4) if raw.get("low", "") != "" else "",
                    "close": round(float(close_raw), 4),
                    "volume": int(float(raw["volume"])) if raw.get("volume", "") != "" else 0,
                    "source": (raw.get("source") or "").strip() or "yahoo_gc_f",
                }
            )
    rows.sort(key=lambda r: r["date"])
    return rows


def fetch_yahoo_gold(period1: int | None = None, period2: int | None = None) -> list[dict]:
    """Fetch daily GC=F OHLCV bars from Yahoo Finance for a full 5y or bounded window."""
    if period1 is not None and period2 is not None:
        url = YAHOO_RANGE_URL.format(period1=period1, period2=period2)
    else:
        url = YAHOO_URL

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    opens = quote.get("open") or [None] * len(timestamps)
    highs = quote.get("high") or [None] * len(timestamps)
    lows = quote.get("low") or [None] * len(timestamps)
    closes = quote.get("close") or [None] * len(timestamps)
    volumes = quote.get("volume") or [None] * len(timestamps)

    rows = []
    for i, ts in enumerate(timestamps):
        close = closes[i] if i < len(closes) else None
        if close is None:
            continue
        d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        open_v = opens[i] if i < len(opens) else None
        high_v = highs[i] if i < len(highs) else None
        low_v = lows[i] if i < len(lows) else None
        vol_v = volumes[i] if i < len(volumes) else None
        rows.append(
            {
                "date": d,
                "open": round(float(open_v), 4) if open_v is not None else "",
                "high": round(float(high_v), 4) if high_v is not None else "",
                "low": round(float(low_v), 4) if low_v is not None else "",
                "close": round(float(close), 4),
                "volume": int(vol_v) if vol_v is not None else 0,
                "source": "yahoo_gc_f",
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


def fetch_yahoo_5y() -> list[dict]:
    """Backwards-compatible wrapper for fetching full 5-year GC=F daily bars."""
    return fetch_yahoo_gold()


def merge_gold_rows(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Merge existing and incoming GC=F rows by date, letting incoming overwrite overlaps."""
    by_date: dict[str, dict] = {r["date"]: r for r in existing}
    for row in incoming:
        by_date[row["date"]] = row
    return [by_date[d] for d in sorted(by_date)]


def save_csv(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def refresh_gold_data(full: bool = False) -> dict:
    """Incrementally fetch latest GC=F bars (with overlap) or rebuild the full 5y history."""
    existing = [] if full else load_existing_rows()

    if existing:
        last_dt = datetime.strptime(existing[-1]["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        period1 = int((last_dt - timedelta(days=OVERLAP_DAYS)).timestamp())
        period2 = int(time.time())
        delta_rows = fetch_yahoo_gold(period1=period1, period2=period2)
        merged = merge_gold_rows(existing, delta_rows)
        mode = "incremental"
        fetched_rows = len(delta_rows)
        added_rows = len(merged) - len(existing)
    else:
        merged = fetch_yahoo_gold()
        if not merged:
            raise ValueError("0 rows returned")
        mode = "full"
        fetched_rows = len(merged)
        added_rows = len(merged)

    if not merged:
        raise ValueError("0 rows returned")

    save_csv(merged)
    closes = [r["close"] for r in merged]
    return {
        "mode": mode,
        "total_rows": len(merged),
        "fetched_rows": fetched_rows,
        "added_rows": added_rows,
        "start_date": merged[0]["date"],
        "end_date": merged[-1]["date"],
        "min_close": min(closes),
        "max_close": max(closes),
        "latest_close": merged[-1]["close"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Gold Futures (GC=F) daily OHLCV data.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force a full 5-year refetch instead of incremental delta updates.",
    )
    args = parser.parse_args(argv)

    try:
        summary = refresh_gold_data(full=args.full)
    except (urllib.error.URLError, KeyError, IndexError, ValueError, OSError) as err:
        print(f"STATUS: FAILED ({err})")
        return 1

    print(f"STATUS: SUCCESS ({summary['mode']})")
    print("Source        : Yahoo Finance (GC=F Gold Futures)")
    print(f"Fetched delta : {summary['fetched_rows']:,} bars ({summary['added_rows']:+,} net new days)")
    print(f"Total rows    : {summary['total_rows']:,} trading days")
    print(f"Date range    : {summary['start_date']} -> {summary['end_date']}")
    print(f"Price min/max : ${summary['min_close']:,.2f} / ${summary['max_close']:,.2f}")
    print(f"Latest close  : ${summary['latest_close']:,.2f} ({summary['end_date']})")
    print(f"Saved CSV     : {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
