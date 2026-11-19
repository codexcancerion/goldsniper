"""
Build and save comprehensive, leak-free forecasting feature datasets (Daily & Weekly)
combining:
  1. Short-, medium-, and long-window Gold technical & regime predictors (lags, EMAs, SMAs, RSI, Bollinger %B, Volatility)
  2. Expanded 12-series Macro & Cross-Asset predictors (DXY, 10Y/5Y/30Y Treasury Yields, Yield Curve Slopes,
     Silver, WTI Oil, Copper, VIX, S&P 500, GDX Gold Miners, EUR/USD, USD/JPY, and Gold Ratios)

Outputs:
  - data/gold_daily_features.csv
  - data/gold_weekly_features.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLD_CSV = DATA_DIR / "gold_daily_5y_clean.csv"
MACRO_CSV = DATA_DIR / "macro_daily_5y.csv"
DAILY_FEAT_CSV = DATA_DIR / "gold_daily_features.csv"
WEEKLY_FEAT_CSV = DATA_DIR / "gold_weekly_features.csv"

MACRO_COLS = [
    "dxy", "us10y", "us5y", "us30y", "silver", "oil",
    "copper", "vix", "sp500", "gdx", "eurusd", "usdjpy",
]


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_daily_features(gold: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    df = gold[["open", "high", "low", "close", "volume", "daily_return"]].join(macro, how="left")
    for col in MACRO_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    c = df["close"]
    ret = df["daily_return"]
    feat = pd.DataFrame(index=df.index)

    # Targets
    feat["close"] = c
    feat["daily_return"] = ret

    # 1. Short-window Gold technicals (all shifted by 1 day -> strictly known at t-1)
    feat["lag_1d"] = c.shift(1)
    feat["ema_3d"] = c.ewm(span=3, adjust=False).mean().shift(1).round(2)
    feat["ema_5d"] = c.ewm(span=5, adjust=False).mean().shift(1).round(2)
    feat["ema_10d"] = c.ewm(span=10, adjust=False).mean().shift(1).round(2)
    feat["ema_20d"] = c.ewm(span=20, adjust=False).mean().shift(1).round(2)

    sma_3 = c.rolling(3).mean().shift(1)
    sma_20 = c.rolling(20).mean().shift(1)
    sma_100 = c.rolling(100, min_periods=20).mean().shift(1)
    ema_200 = c.ewm(span=200, adjust=False).mean().shift(1)
    ema_500 = c.ewm(span=500, adjust=False).mean().shift(1)

    feat["sma_3d_spread"] = ((c.shift(1) / sma_3) - 1.0).round(6)
    feat["ema_5d_vs_10d"] = ((feat["ema_5d"] / feat["ema_10d"]) - 1.0).round(6)
    feat["sma_20d_vs_100d"] = ((sma_20 / sma_100) - 1.0).round(6)
    feat["ema_200d_vs_500d"] = ((ema_200 / ema_500) - 1.0).round(6)
    feat["close_vs_ema20d"] = ((c.shift(1) / feat["ema_20d"]) - 1.0).round(6)
    feat["close_vs_ema200d"] = ((c.shift(1) / ema_200) - 1.0).round(6)

    feat["lag_1d_return"] = ret.shift(1).round(6)
    feat["lag_2d_return"] = ret.shift(2).round(6)
    feat["lag_3d_return"] = ret.shift(3).round(6)
    feat["lag_5d_return"] = c.pct_change(5).shift(1).round(6)
    feat["lag_7d_return"] = ret.shift(7).round(6)
    feat["lag_21d_return"] = c.pct_change(21).shift(1).round(6)

    # Volatility, RSI, Bollinger %B, Intraday Range (t-1)
    std_20 = c.rolling(20).std().shift(1)
    feat["vol_5d"] = ret.rolling(5).std().shift(1).round(6)
    feat["vol_20d"] = ret.rolling(20).std().shift(1).round(6)
    feat["hl_range_lag1"] = (((df["high"] - df["low"]) / c).shift(1)).round(6)
    feat["rsi_14d_lag1"] = (compute_rsi(c, 14).shift(1) / 100.0).round(4)
    feat["bb_pct_b_lag1"] = (((c.shift(1) - (sma_20 - 2 * std_20)) / (4 * std_20)).clip(-1, 2)).round(4)

    # 2. Expanded 12-Macro levels, changes, EMA spreads, and cross-asset ratios (t-1)
    for col in ["dxy", "us10y", "us5y", "us30y", "silver", "oil", "copper", "vix", "sp500", "gdx", "eurusd", "usdjpy"]:
        feat[f"{col}_lag1"] = df[col].shift(1).round(4)
        if col in ("us10y", "us5y", "us30y"):
            feat[f"{col}_chg_lag1"] = df[col].diff().shift(1).round(4)
        else:
            feat[f"{col}_ret_lag1"] = df[col].pct_change().shift(1).round(6)
        col_ema20 = df[col].ewm(span=20, adjust=False).mean().shift(1)
        feat[f"{col}_vs_ema20d"] = ((df[col].shift(1) / col_ema20) - 1.0).round(6)

    feat["oil_mom_20d"] = df["oil"].pct_change(20).shift(1).round(6)
    feat["dxy_mom_20d"] = df["dxy"].pct_change(20).shift(1).round(6)

    # Ratios & Yield Curve Slopes (t-1)
    gs_ratio = c / df["silver"]
    gg_ratio = c / df["gdx"]
    gc_ratio = c / df["copper"]
    feat["gold_silver_ratio_lag1"] = gs_ratio.shift(1).round(4)
    feat["gold_gdx_ratio_lag1"] = gg_ratio.shift(1).round(4)
    feat["gold_gdx_ret_lag1"] = gg_ratio.pct_change().shift(1).round(6)
    feat["gold_gdx_z5_lag1"] = ((gg_ratio.shift(1) / gg_ratio.rolling(5).mean().shift(1)) - 1.0).round(6)
    feat["gold_copper_ratio_lag1"] = gc_ratio.shift(1).round(4)
    feat["yc_10y_5y_lag1"] = (df["us10y"] - df["us5y"]).shift(1).round(4)
    feat["yc_30y_10y_lag1"] = (df["us30y"] - df["us10y"]).shift(1).round(4)

    feat = feat.iloc[25:].dropna().copy()
    return feat


def build_weekly_features(gold: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    merged = gold[["close", "high", "low"]].join(macro, how="left")
    for col in MACRO_COLS:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()

    wk = pd.DataFrame(
        {
            "close": merged["close"].resample("W-FRI").last(),
            "high": merged["high"].resample("W-FRI").max(),
            "low": merged["low"].resample("W-FRI").min(),
            **{col: merged[col].resample("W-FRI").last() for col in MACRO_COLS},
        }
    ).dropna()

    c = wk["close"]
    ret = c.pct_change()
    feat = pd.DataFrame(index=wk.index)

    # Targets
    feat["close"] = c.round(2)
    feat["weekly_return"] = ret.round(6)

    # 1. Weekly Gold technicals (strictly shifted by 1 week -> known at t-1w)
    feat["lag_1w"] = c.shift(1).round(2)
    feat["ema_2w"] = c.ewm(span=2, adjust=False).mean().shift(1).round(2)
    feat["ema_4w"] = c.ewm(span=4, adjust=False).mean().shift(1).round(2)
    feat["ema_8w"] = c.ewm(span=8, adjust=False).mean().shift(1).round(2)
    sma_20w = c.rolling(20, min_periods=8).mean().shift(1)
    ema_26w = c.ewm(span=26, adjust=False).mean().shift(1)
    ema_104w = c.ewm(span=104, adjust=False).mean().shift(1)

    feat["ema_2w_vs_4w"] = ((feat["ema_2w"] / feat["ema_4w"]) - 1.0).round(6)
    feat["ema_4w_vs_8w"] = ((feat["ema_4w"] / feat["ema_8w"]) - 1.0).round(6)
    feat["close_vs_ema4w"] = ((c.shift(1) / feat["ema_4w"]) - 1.0).round(6)
    feat["close_vs_sma20w"] = ((c.shift(1) / sma_20w) - 1.0).round(6)
    feat["ema_26w_vs_104w"] = ((ema_26w / ema_104w) - 1.0).round(6)

    feat["lag_1w_return"] = ret.shift(1).round(6)
    feat["lag_2w_return"] = ret.shift(2).round(6)
    feat["lag_4w_return"] = ret.shift(4).round(6)
    feat["lag_6w_return"] = ret.shift(6).round(6)
    feat["lag_12w_return"] = ret.shift(12).round(6)
    feat["mom_4w"] = c.pct_change(4).shift(1).round(6)

    feat["vol_4w"] = ret.rolling(4).std().shift(1).round(6)
    feat["hl_range_lag1w"] = (((wk["high"] - wk["low"]) / c).shift(1)).round(6)
    feat["rsi_14w_lag1w"] = (compute_rsi(c, 14).shift(1) / 100.0).round(4)

    # 2. Weekly Macro predictors (t-1w)
    for col in MACRO_COLS:
        feat[f"{col}_lag1w"] = wk[col].shift(1).round(4)
        if col in ("us10y", "us5y", "us30y"):
            feat[f"{col}_chg_lag1w"] = wk[col].diff().shift(1).round(4)
        else:
            feat[f"{col}_ret_lag1w"] = wk[col].pct_change().shift(1).round(6)
        col_ema8 = wk[col].ewm(span=8, adjust=False).mean().shift(1)
        feat[f"{col}_vs_ema8w"] = ((wk[col].shift(1) / col_ema8) - 1.0).round(6)

    feat["oil_mom_8w"] = wk["oil"].pct_change(8).shift(1).round(6)
    feat["dxy_mom_8w"] = wk["dxy"].pct_change(8).shift(1).round(6)
    feat["silver_chg_lag6w"] = wk["silver"].pct_change().shift(6).round(6)
    feat["dxy_chg_lag12w"] = wk["dxy"].pct_change().shift(12).round(6)

    gs_ratio = c / wk["silver"]
    gg_ratio = c / wk["gdx"]
    gc_ratio = c / wk["copper"]
    feat["gold_silver_ratio_lag1w"] = gs_ratio.shift(1).round(4)
    feat["gold_gdx_ratio_lag1w"] = gg_ratio.shift(1).round(4)
    feat["gold_gdx_ret_lag1w"] = gg_ratio.pct_change().shift(1).round(6)
    feat["gold_copper_ratio_lag1w"] = gc_ratio.shift(1).round(4)
    feat["yc_10y_5y_lag1w"] = (wk["us10y"] - wk["us5y"]).shift(1).round(4)
    feat["yc_30y_10y_lag1w"] = (wk["us30y"] - wk["us10y"]).shift(1).round(4)

    feat = feat.iloc[13:].dropna().copy()
    return feat


def main() -> None:
    gold = pd.read_csv(GOLD_CSV, parse_dates=["date"]).set_index("date").sort_index()
    macro = pd.read_csv(MACRO_CSV, parse_dates=["date"]).set_index("date").sort_index()

    daily_feat = build_daily_features(gold, macro)
    weekly_feat = build_weekly_features(gold, macro)

    daily_out = daily_feat.reset_index()
    daily_out["date"] = daily_out["date"].dt.strftime("%Y-%m-%d")
    daily_out.to_csv(DAILY_FEAT_CSV, index=False)

    weekly_out = weekly_feat.reset_index()
    weekly_out["date"] = weekly_out["date"].dt.strftime("%Y-%m-%d")
    weekly_out.to_csv(WEEKLY_FEAT_CSV, index=False)

    print("EXPANDED FEATURE BUILD STATUS: SUCCESS")
    print(f"Daily features  : {len(daily_out):,} rows x {daily_out.shape[1]} cols ({daily_out['date'].iloc[0]} -> {daily_out['date'].iloc[-1]})")
    print(f"Weekly features : {len(weekly_out):,} rows x {weekly_out.shape[1]} cols ({weekly_out['date'].iloc[0]} -> {weekly_out['date'].iloc[-1]})")


if __name__ == "__main__":
    main()
