"""
Clean the 5-year daily Gold Futures (GC=F) dataset and export two forecast-ready CSVs:
  1. data/gold_daily_5y_clean.csv : Settled trading days (for Prophet, ML, EDA)
  2. data/gold_daily_5y_bday.csv  : Uniform business-day frequency with holidays forward-filled
                                    (for ARIMA, SARIMA, Holt-Winters / ETS)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "gold_daily_5y.csv"
CLEAN_CSV = DATA_DIR / "gold_daily_5y_clean.csv"
BDAY_CSV = DATA_DIR / "gold_daily_5y_bday.csv"


def clean_gold_data() -> None:
    df = pd.read_csv(RAW_CSV)
    raw_rows = len(df)

    # 1. Parse dates, sort chronologically, remove any duplicates
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # 2. Drop today's incomplete/live intraday bar (keep only settled daily closes)
    today = pd.Timestamp.today().normalize()
    dropped_live = int((df["date"] >= today).sum())
    df = df[df["date"] < today].copy()

    # 3. Drop constant provenance column
    if "source" in df.columns:
        df = df.drop(columns=["source"])

    # 4. Fix IEEE-754 floating-point noise in OHLC prices
    ohlc_cols = ["open", "high", "low", "close"]
    df[ohlc_cols] = df[ohlc_cols].round(2)

    # 5. Impute zero-volume anomalies with 21-trading-day rolling median
    zero_vol_mask = df["volume"] <= 0
    imputed_vol_count = int(zero_vol_mask.sum())
    df["volume_imputed"] = zero_vol_mask
    vol_series = df["volume"].mask(zero_vol_mask, np.nan)
    rolling_med = vol_series.rolling(window=21, min_periods=1, center=True).median()
    df["volume"] = vol_series.fillna(rolling_med).ffill().bfill().round().astype(int)

    # 6. Add variance-stabilized & return features for time-series modeling
    df["log_close"] = np.log(df["close"]).round(6)
    df["daily_return"] = df["close"].pct_change().round(6)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)).round(6)

    # Save trading-day clean dataset
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(CLEAN_CSV, index=False)

    # 7. Create uniform Business-Day ('B') dataset for ARIMA / ETS models
    bday = df.set_index("date").asfreq("B")
    holiday_mask = bday["close"].isna()
    holidays_filled = int(holiday_mask.sum())

    # Forward-fill prices on market holidays; set holiday volume to 0 and mark flag
    bday[ohlc_cols] = bday[ohlc_cols].ffill()
    bday["volume"] = bday["volume"].fillna(0).astype(int)
    bday["volume_imputed"] = bday["volume_imputed"].eq(True)
    bday["is_holiday_filled"] = holiday_mask
    bday["log_close"] = np.log(bday["close"]).round(6)
    bday["daily_return"] = bday["close"].pct_change().round(6)
    bday["log_return"] = np.log(bday["close"] / bday["close"].shift(1)).round(6)

    bday_out = bday.reset_index()
    bday_out["date"] = bday_out["date"].dt.strftime("%Y-%m-%d")
    bday_out.to_csv(BDAY_CSV, index=False)

    # Verification summary
    print("CLEANING STATUS: SUCCESS")
    print(f"Raw rows loaded         : {raw_rows:,}")
    print(f"Live intraday dropped   : {dropped_live}")
    print(f"Zero-volume imputed     : {imputed_vol_count}")
    print(f"Trading-day clean rows  : {len(df_out):,} ({df_out['date'].iloc[0]} -> {df_out['date'].iloc[-1]})")
    print(f"Business-day rows       : {len(bday_out):,} ({holidays_filled} market holidays forward-filled)")
    print(f"Nulls in OHLC (clean)   : {int(df[ohlc_cols].isna().sum().sum())}")
    print(f"Nulls in OHLC (bday)    : {int(bday[ohlc_cols].isna().sum().sum())}")
    print(f"Saved trading-day CSV   : {CLEAN_CSV}")
    print(f"Saved business-day CSV  : {BDAY_CSV}")


if __name__ == "__main__":
    clean_gold_data()
