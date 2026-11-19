"""
Explore predictor correlations for 5-year Gold Futures (GC=F) forecasting:
  - Daily  : Lag days, Daily SMAs, Daily EMAs (shifted by 1 day to prevent leakage)
  - Weekly : Lag weeks, Weekly SMAs, Weekly EMAs (shifted by 1 week to prevent leakage)

Evaluates BOTH:
  1. Price-Level correlation (Pearson r & tracking RMSE against close_t)
  2. Stationary Return correlation (Pearson r, p-value, Spearman rho against return_t)
     using lagged returns and lagged price-to-MA/EMA spreads (momentum vs mean-reversion).
  3. PACF (Partial Autocorrelation) to isolate independent lag signals.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "gold_daily_5y_clean.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_timeframe(
    series: pd.Series,
    lags: list[int],
    windows: list[int],
    unit_label: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build leak-free lag, SMA, and EMA predictors and compute correlations."""
    df = pd.DataFrame({"close": series})
    df["ret"] = df["close"].pct_change()

    records: list[dict] = []

    # 1. Lags
    for k in lags:
        lag_price = df["close"].shift(k)
        lag_ret = df["ret"].shift(k)

        valid_p = pd.concat([df["close"], lag_price], axis=1).dropna()
        r_price = valid_p.iloc[:, 0].corr(valid_p.iloc[:, 1])
        rmse_price = float(np.sqrt(np.mean((valid_p.iloc[:, 0] - valid_p.iloc[:, 1]) ** 2)))

        valid_r = pd.concat([df["ret"], lag_ret], axis=1).dropna()
        r_ret, p_ret = stats.pearsonr(valid_r.iloc[:, 0], valid_r.iloc[:, 1])
        rho_ret, _ = stats.spearmanr(valid_r.iloc[:, 0], valid_r.iloc[:, 1])

        records.append(
            {
                "type": f"Lag ({unit_label})",
                "window": k,
                "feature": f"lag_{k}{unit_label[0].lower()}",
                "price_pearson_r": round(float(r_price), 6),
                "price_rmse": round(rmse_price, 2),
                "return_pearson_r": round(float(r_ret), 4),
                "return_p_value": round(float(p_ret), 4),
                "return_spearman_rho": round(float(rho_ret), 4),
                "return_significant_5pct": bool(p_ret < 0.05),
                "n_obs": len(valid_r),
            }
        )

    # 2. Simple Moving Averages (shifted by 1 to avoid look-ahead leakage)
    for w in windows:
        sma_prev = df["close"].rolling(window=w, min_periods=w).mean().shift(1)
        # Spread: how far yesterday's close was above/below yesterday's SMA
        sma_spread = (df["close"].shift(1) / sma_prev) - 1.0

        valid_p = pd.concat([df["close"], sma_prev], axis=1).dropna()
        r_price = valid_p.iloc[:, 0].corr(valid_p.iloc[:, 1])
        rmse_price = float(np.sqrt(np.mean((valid_p.iloc[:, 0] - valid_p.iloc[:, 1]) ** 2)))

        valid_r = pd.concat([df["ret"], sma_spread], axis=1).dropna()
        r_ret, p_ret = stats.pearsonr(valid_r.iloc[:, 0], valid_r.iloc[:, 1])
        rho_ret, _ = stats.spearmanr(valid_r.iloc[:, 0], valid_r.iloc[:, 1])

        records.append(
            {
                "type": f"SMA ({unit_label})",
                "window": w,
                "feature": f"sma_{w}{unit_label[0].lower()}",
                "price_pearson_r": round(float(r_price), 6),
                "price_rmse": round(rmse_price, 2),
                "return_pearson_r": round(float(r_ret), 4),
                "return_p_value": round(float(p_ret), 4),
                "return_spearman_rho": round(float(rho_ret), 4),
                "return_significant_5pct": bool(p_ret < 0.05),
                "n_obs": len(valid_r),
            }
        )

    # 3. Exponential Moving Averages (shifted by 1 to avoid look-ahead leakage)
    for w in windows:
        ema_prev = df["close"].ewm(span=w, adjust=False).mean().shift(1)
        # Burn-in first w rows for fair comparison with SMA
        ema_prev.iloc[:w] = np.nan
        ema_spread = (df["close"].shift(1) / ema_prev) - 1.0

        valid_p = pd.concat([df["close"], ema_prev], axis=1).dropna()
        r_price = valid_p.iloc[:, 0].corr(valid_p.iloc[:, 1])
        rmse_price = float(np.sqrt(np.mean((valid_p.iloc[:, 0] - valid_p.iloc[:, 1]) ** 2)))

        valid_r = pd.concat([df["ret"], ema_spread], axis=1).dropna()
        r_ret, p_ret = stats.pearsonr(valid_r.iloc[:, 0], valid_r.iloc[:, 1])
        rho_ret, _ = stats.spearmanr(valid_r.iloc[:, 0], valid_r.iloc[:, 1])

        records.append(
            {
                "type": f"EMA ({unit_label})",
                "window": w,
                "feature": f"ema_{w}{unit_label[0].lower()}",
                "price_pearson_r": round(float(r_price), 6),
                "price_rmse": round(rmse_price, 2),
                "return_pearson_r": round(float(r_ret), 4),
                "return_p_value": round(float(p_ret), 4),
                "return_spearman_rho": round(float(rho_ret), 4),
                "return_significant_5pct": bool(p_ret < 0.05),
                "n_obs": len(valid_r),
            }
        )

    res_df = pd.DataFrame(records)

    # PACF on stationary returns (lags 1..20)
    ret_clean = df["ret"].dropna()
    max_lag = min(20, len(ret_clean) // 5)
    pacf_vals = pacf(ret_clean, nlags=max_lag, method="ywm")
    pacf_series = pd.Series(pacf_vals[1:], index=range(1, max_lag + 1), name="pacf_return")

    return res_df, pacf_series


def plot_overview(daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Gold (GC=F) Predictor Exploration: Lags, SMAs & EMAs (5-Year Daily & Weekly)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Daily Price Tracking RMSE (lower = stronger level tracking)
    ax = axes[0, 0]
    for ftype, color, marker in [
        ("Lag (Days)", "#1f77b4", "o"),
        ("SMA (Days)", "#ff7f0e", "s"),
        ("EMA (Days)", "#2ca02c", "^"),
    ]:
        sub = daily_df[(daily_df["type"] == ftype) & (daily_df["window"] <= 60)]
        ax.plot(sub["window"], sub["price_rmse"], label=ftype, color=color, marker=marker, linewidth=2)
    ax.set_title("Daily Price Tracking Error (RMSE vs Close_t, Lower = Better)")
    ax.set_xlabel("Lookback Window (Trading Days)")
    ax.set_ylabel("RMSE (USD)")
    ax.legend()

    # Panel 2: Daily Return Correlations (top 12 by |r|)
    ax = axes[0, 1]
    top_d = daily_df.reindex(daily_df["return_pearson_r"].abs().sort_values(ascending=False).index).head(12)
    top_d = top_d.iloc[::-1]
    colors = ["#2ca02c" if sig else "#95a5a6" for sig in top_d["return_significant_5pct"]]
    ax.barh(top_d["feature"], top_d["return_pearson_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    crit_d = 1.96 / np.sqrt(daily_df["n_obs"].median())
    ax.axvline(crit_d, color="red", linestyle="--", linewidth=1, label=f"95% threshold (±{crit_d:.3f})")
    ax.axvline(-crit_d, color="red", linestyle="--", linewidth=1)
    ax.set_title("Daily Return Predictors (Top 12 by |r|; Green = p < 0.05)")
    ax.set_xlabel("Pearson r with Next-Day Return")
    ax.legend(loc="lower right", fontsize=8)

    # Panel 3: Weekly Price Tracking RMSE
    ax = axes[1, 0]
    for ftype, color, marker in [
        ("Lag (Weeks)", "#1f77b4", "o"),
        ("SMA (Weeks)", "#ff7f0e", "s"),
        ("EMA (Weeks)", "#2ca02c", "^"),
    ]:
        sub = weekly_df[(weekly_df["type"] == ftype) & (weekly_df["window"] <= 26)]
        ax.plot(sub["window"], sub["price_rmse"], label=ftype, color=color, marker=marker, linewidth=2)
    ax.set_title("Weekly Price Tracking Error (RMSE vs Close_t, Lower = Better)")
    ax.set_xlabel("Lookback Window (Weeks)")
    ax.set_ylabel("RMSE (USD)")
    ax.legend()

    # Panel 4: Weekly Return Correlations (top 12 by |r|)
    ax = axes[1, 1]
    top_w = weekly_df.reindex(weekly_df["return_pearson_r"].abs().sort_values(ascending=False).index).head(12)
    top_w = top_w.iloc[::-1]
    colors = ["#2ca02c" if sig else "#95a5a6" for sig in top_w["return_significant_5pct"]]
    ax.barh(top_w["feature"], top_w["return_pearson_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    crit_w = 1.96 / np.sqrt(weekly_df["n_obs"].median())
    ax.axvline(crit_w, color="red", linestyle="--", linewidth=1, label=f"95% threshold (±{crit_w:.3f})")
    ax.axvline(-crit_w, color="red", linestyle="--", linewidth=1)
    ax.set_title("Weekly Return Predictors (Top 12 by |r|; Green = p < 0.05)")
    ax.set_xlabel("Pearson r with Next-Week Return")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "predictor_correlation_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
    daily_close = df["close"]
    weekly_close = daily_close.resample("W-FRI").last().dropna()

    print(f"Daily observations : {len(daily_close):,} ({daily_close.index[0].date()} -> {daily_close.index[-1].date()})")
    print(f"Weekly observations: {len(weekly_close):,} ({weekly_close.index[0].date()} -> {weekly_close.index[-1].date()})")

    daily_lags = [1, 2, 3, 4, 5, 7, 10, 14, 20, 21, 30, 60]
    daily_windows = [3, 5, 7, 10, 14, 20, 21, 30, 50, 100, 200]
    daily_res, daily_pacf = evaluate_timeframe(daily_close, daily_lags, daily_windows, "Days")

    weekly_lags = [1, 2, 3, 4, 6, 8, 12, 16, 20, 26, 52]
    weekly_windows = [2, 3, 4, 6, 8, 10, 12, 16, 20, 26, 52]
    weekly_res, weekly_pacf = evaluate_timeframe(weekly_close, weekly_lags, weekly_windows, "Weeks")

    daily_csv = OUT_DIR / "daily_predictor_correlations.csv"
    weekly_csv = OUT_DIR / "weekly_predictor_correlations.csv"
    daily_res.to_csv(daily_csv, index=False)
    weekly_res.to_csv(weekly_csv, index=False)

    chart_path = plot_overview(daily_res, weekly_res)

    print("\n=== TOP DAILY PRICE-LEVEL PREDICTORS (by Lowest RMSE / Highest r) ===")
    for ftype in ["Lag (Days)", "SMA (Days)", "EMA (Days)"]:
        top = daily_res[daily_res["type"] == ftype].sort_values("price_rmse").head(5)
        print(f"\n-- {ftype} --")
        print(top[["feature", "price_pearson_r", "price_rmse"]].to_string(index=False))

    print("\n=== TOP DAILY RETURN PREDICTORS (Stationary Signal, Ranked by |r|) ===")
    top_d_ret = daily_res.reindex(daily_res["return_pearson_r"].abs().sort_values(ascending=False).index).head(10)
    print(top_d_ret[["feature", "type", "return_pearson_r", "return_p_value", "return_spearman_rho", "return_significant_5pct"]].to_string(index=False))

    crit_d = 1.96 / np.sqrt(len(daily_close) - 1)
    sig_d_pacf = daily_pacf[daily_pacf.abs() > crit_d]
    print(f"\nSignificant Daily Return PACF Lags (|PACF| > {crit_d:.3f}):")
    print(sig_d_pacf.round(4).to_string() if len(sig_d_pacf) else "  None above 95% CI")

    print("\n=== TOP WEEKLY PRICE-LEVEL PREDICTORS (by Lowest RMSE / Highest r) ===")
    for ftype in ["Lag (Weeks)", "SMA (Weeks)", "EMA (Weeks)"]:
        top = weekly_res[weekly_res["type"] == ftype].sort_values("price_rmse").head(5)
        print(f"\n-- {ftype} --")
        print(top[["feature", "price_pearson_r", "price_rmse"]].to_string(index=False))

    print("\n=== TOP WEEKLY RETURN PREDICTORS (Stationary Signal, Ranked by |r|) ===")
    top_w_ret = weekly_res.reindex(weekly_res["return_pearson_r"].abs().sort_values(ascending=False).index).head(10)
    print(top_w_ret[["feature", "type", "return_pearson_r", "return_p_value", "return_spearman_rho", "return_significant_5pct"]].to_string(index=False))

    crit_w = 1.96 / np.sqrt(len(weekly_close) - 1)
    sig_w_pacf = weekly_pacf[weekly_pacf.abs() > crit_w]
    print(f"\nSignificant Weekly Return PACF Lags (|PACF| > {crit_w:.3f}):")
    print(sig_w_pacf.round(4).to_string() if len(sig_w_pacf) else "  None above 95% CI")

    print(f"\nSaved Daily CSV  : {daily_csv}")
    print(f"Saved Weekly CSV : {weekly_csv}")
    print(f"Saved Chart      : {chart_path}")


if __name__ == "__main__":
    main()
