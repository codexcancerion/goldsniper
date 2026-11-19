"""
Fetch the last 5 years of daily Gold Futures (GC=F) OHLCV data
from Yahoo Finance and save to data/gold_daily_5y.csv.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=5y&interval=1d"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = OUTPUT_DIR / "gold_daily_5y.csv"


def fetch_yahoo_5y() -> list[dict]:
    req = urllib.request.Request(
        YAHOO_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    rows = []
    for i, ts in enumerate(timestamps):
        close = quote["close"][i]
        if close is None:
            continue
        d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append(
            {
                "date": d,
                "open": round(float(quote["open"][i]), 4) if quote["open"][i] is not None else "",
                "high": round(float(quote["high"][i]), 4) if quote["high"][i] is not None else "",
                "low": round(float(quote["low"][i]), 4) if quote["low"][i] is not None else "",
                "close": round(float(close), 4),
                "volume": int(quote["volume"][i]) if quote["volume"][i] is not None else 0,
                "source": "yahoo_gc_f",
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


def save_csv(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    try:
        rows = fetch_yahoo_5y()
    except (urllib.error.URLError, KeyError, IndexError, ValueError) as err:
        print(f"STATUS: FAILED ({err})")
        return 1

    if not rows:
        print("STATUS: FAILED (0 rows returned)")
        return 1

    save_csv(rows)

    closes = [r["close"] for r in rows]
    print("STATUS: SUCCESS")
    print("Source        : Yahoo Finance (GC=F Gold Futures)")
    print(f"Total rows    : {len(rows):,} trading days")
    print(f"Date range    : {rows[0]['date']} -> {rows[-1]['date']}")
    print(f"Price min/max : ${min(closes):,.2f} / ${max(closes):,.2f}")
    print(f"Latest close  : ${rows[-1]['close']:,.2f} ({rows[-1]['date']})")
    print(f"Saved CSV     : {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
