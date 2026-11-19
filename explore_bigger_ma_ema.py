"""
Explore larger SMA & EMA windows (medium- to long-term trend/regime indicators)
for 5-year Gold Futures (GC=F):
  - Daily windows : 30, 50, 75, 100, 125, 150, 200, 250, 300, 365, 400, 500 trading days
  - Weekly windows: 12, 20, 26, 30, 40, 52, 65, 78, 104 weeks

Evaluates leak-free (t-1 shifted) indicators against:
  1. Price level (close_t): Pearson r & RMSE
  2. Multi-horizon forward returns:
     - Daily  : 1-day, 5-day (~1w), 21-day (~1m), 63-day (~1q) forward returns
     - Weekly : 1-week, 4-week (~1m), 12-week (~1q), 26-week (~6m) forward returns
  3. Classic long-term MA/EMA crossover ratios (e.g., 50/200, 100/300, 20w/52w)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "gold_daily_5y_clean.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_big_windows(
    series: pd.Series,
    windows: list[int],
    horizons: list[int],
    unit_label: str,
) -> pd.DataFrame:
    df = pd.DataFrame({"close": series})
    # Forward returns from t-1 to t-1+h (strictly aligned so feature at t-1 predicts future return)
    # Specifically: feature_prev is known at close of t-1; target is return over next h periods
    fwd_rets = {}
    for h in horizons:
        fwd_rets[h] = (df["close"].shift(-(h - 1)) / df["close"].shift(1)) - 1.0

    rows: list[dict] = []
    u = unit_label[0].lower()

    for kind in ["SMA", "EMA"]:
        for w in windows:
            if kind == "SMA":
                ma_prev = df["close"].rolling(window=w, min_periods=w).mean().shift(1)
            else:
                ma_prev = df["close"].ewm(span=w, adjust=False).mean().shift(1)
                ma_prev.iloc[:w] = np.nan

            spread_prev = (df["close"].shift(1) / ma_prev) - 1.0

            valid_p = pd.concat([df["close"], ma_prev], axis=1).dropna()
            r_price = valid_p.iloc[:, 0].corr(valid_p.iloc[:, 1])
            rmse_price = float(np.sqrt(np.mean((valid_p.iloc[:, 0] - valid_p.iloc[:, 1]) ** 2)))

            rec: dict = {
                "timeframe": unit_label,
                "type": kind,
                "window": w,
                "feature": f"{kind.lower()}_{w}{u}",
                "price_pearson_r": round(float(r_price), 5),
                "price_rmse": round(rmse_price, 2),
                "n_obs": len(valid_p),
            }

            for h in horizons:
                valid_r = pd.concat([fwd_rets[h], spread_prev], axis=1).dropna()
                if len(valid_r) >= 20:
                    r_val, p_val = stats.pearsonr(valid_r.iloc[:, 0], valid_r.iloc[:, 1])
                    rec[f"r_fwd_{h}{u}"] = round(float(r_val), 4)
                    rec[f"p_fwd_{h}{u}"] = round(float(p_val), 4)
                else:
                    rec[f"r_fwd_{h}{u}"] = np.nan
                    rec[f"p_fwd_{h}{u}"] = np.nan

            rows.append(rec)

    return pd.DataFrame(rows)


def evaluate_crossovers(
    series: pd.Series,
    pairs: list[tuple[int, int]],
    horizons: list[int],
    unit_label: str,
) -> pd.DataFrame:
    df = pd.DataFrame({"close": series})
    fwd_rets = {h: (df["close"].shift(-(h - 1)) / df["close"].shift(1)) - 1.0 for h in horizons}
    u = unit_label[0].lower()
    rows: list[dict] = []

    for kind in ["SMA", "EMA"]:
        for fast, slow in pairs:
            if kind == "SMA":
                ma_f = df["close"].rolling(fast, min_periods=fast).mean().shift(1)
                ma_s = df["close"].rolling(slow, min_periods=slow).mean().shift(1)
            else:
                ma_f = df["close"].ewm(span=fast, adjust=False).mean().shift(1)
                ma_s = df["close"].ewm(span=slow, adjust=False).mean().shift(1)
                ma_s.iloc[:slow] = np.nan

            ratio = (ma_f / ma_s) - 1.0
            rec: dict = {
                "timeframe": unit_label,
                "type": f"{kind}_cross",
                "feature": f"{kind.lower()}_{fast}{u}_vs_{slow}{u}",
                "n_obs": int(ratio.dropna().shape[0]),
            }
            for h in horizons:
                valid = pd.concat([fwd_rets[h], ratio], axis=1).dropna()
                r_val, p_val = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
                rec[f"r_fwd_{h}{u}"] = round(float(r_val), 4)
                rec[f"p_fwd_{h}{u}"] = round(float(p_val), 4)
            rows.append(rec)

    return pd.DataFrame(rows)


def plot_heatmaps(daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        "Bigger SMA & EMA Spreads vs. Multi-Horizon Forward Returns (Pearson r)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Daily heatmap
    d_cols = ["r_fwd_1d", "r_fwd_5d", "r_fwd_21d", "r_fwd_63d"]
    d_mat = daily_df.set_index("feature")[d_cols]
    d_mat.columns = ["1-Day", "5-Day (1W)", "21-Day (1M)", "63-Day (1Q)"]
    sns.heatmap(
        d_mat,
        ax=axes[0],
        cmap="RdBu_r",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Pearson r"},
    )
    axes[0].set_title("Daily: Close/MA Spread vs. Forward Return")
    axes[0].set_ylabel("Indicator (t-1)")

    # Weekly heatmap
    w_cols = ["r_fwd_1w", "r_fwd_4w", "r_fwd_12w", "r_fwd_26w"]
    w_mat = weekly_df.set_index("feature")[w_cols]
    w_mat.columns = ["1-Week", "4-Week (1M)", "12-Week (1Q)", "26-Week (6M)"]
    sns.heatmap(
        w_mat,
        ax=axes[1],
        cmap="RdBu_r",
        center=0,
        vmin=-0.6,
        vmax=0.6,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Pearson r"},
    )
    axes[1].set_title("Weekly: Close/MA Spread vs. Forward Return")
    axes[1].set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "bigger_ma_ema_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
    daily_close = df["close"]
    weekly_close = daily_close.resample("W-FRI").last().dropna()

    daily_windows = [30, 50, 75, 100, 125, 150, 200, 250, 300, 365, 400, 500]
    daily_horizons = [1, 5, 21, 63]
    daily_big = evaluate_big_windows(daily_close, daily_windows, daily_horizons, "Days")

    weekly_windows = [12, 20, 26, 30, 40, 52, 65, 78, 104]
    weekly_horizons = [1, 4, 12, 26]
    weekly_big = evaluate_big_windows(weekly_close, weekly_windows, weekly_horizons, "Weeks")

    daily_cross = evaluate_crossovers(
        daily_close, [(20, 100), (50, 200), (100, 300), (150, 400), (200, 500)], daily_horizons, "Days"
    )
    weekly_cross = evaluate_crossovers(
        weekly_close, [(10, 40), (12, 52), (20, 52), (26, 78), (26, 104)], weekly_horizons, "Weeks"
    )

    daily_big.to_csv(OUT_DIR / "daily_bigger_ma_ema.csv", index=False)
    weekly_big.to_csv(OUT_DIR / "weekly_bigger_ma_ema.csv", index=False)
    chart_path = plot_heatmaps(daily_big, weekly_big)

    print("=== DAILY BIGGER SMA & EMA (30d to 500d) ===")
    print(daily_big[["feature", "price_pearson_r", "price_rmse", "r_fwd_1d", "r_fwd_5d", "r_fwd_21d", "r_fwd_63d", "p_fwd_63d"]].to_string(index=False))

    print("\n=== DAILY CROSSOVER RATIOS (Fast / Slow - 1) ===")
    print(daily_cross[["feature", "r_fwd_1d", "r_fwd_5d", "r_fwd_21d", "r_fwd_63d", "p_fwd_63d"]].to_string(index=False))

    print("\n=== WEEKLY BIGGER SMA & EMA (12w to 104w) ===")
    print(weekly_big[["feature", "price_pearson_r", "price_rmse", "r_fwd_1w", "r_fwd_4w", "r_fwd_12w", "r_fwd_26w", "p_fwd_26w"]].to_string(index=False))

    print("\n=== WEEKLY CROSSOVER RATIOS (Fast / Slow - 1) ===")
    print(weekly_cross[["feature", "r_fwd_1w", "r_fwd_4w", "r_fwd_12w", "r_fwd_26w", "p_fwd_26w"]].to_string(index=False))

    print(f"\nSaved Heatmap: {chart_path}")


if __name__ == "__main__":
    main()
