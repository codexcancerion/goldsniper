"""
Deep-dive macro correlation analysis against Gold (GC=F) on Daily and Weekly timeframes:
  1. Contemporaneous correlations (Level & Return/Change: Pearson r, Spearman rho, p-value)
  2. Lead-Lag cross-correlations (does Macro at t-k predict Gold return at t?)
  3. Macro trend/ratio predictors (Macro EMA spreads & Gold/Silver Ratio vs forward Gold returns)
  4. Granger Causality tests (Daily & Weekly stationary returns/changes)
  5. Rolling correlation stability (60-day daily & 26-week weekly rolling return correlation)

Outputs:
  - outputs/macro_daily_correlations.csv
  - outputs/macro_weekly_correlations.csv
  - outputs/macro_correlation_analysis.png
"""

from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
GOLD_CSV = BASE_DIR / "data" / "gold_daily_5y_clean.csv"
MACRO_CSV = BASE_DIR / "data" / "macro_daily_5y.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MACRO_COLS = {
    "silver": ("Silver (SI=F)", "pct"),
    "dxy": ("USD Index (DXY)", "pct"),
    "us10y": ("10Y Yield (^TNX)", "diff"),
    "oil": ("WTI Oil (CL=F)", "pct"),
}


def compute_changes(df: pd.DataFrame) -> pd.DataFrame:
    chg = pd.DataFrame(index=df.index)
    chg["gold"] = df["gold"].pct_change()
    for col, (_, mode) in MACRO_COLS.items():
        chg[col] = df[col].diff() if mode == "diff" else df[col].pct_change()
    chg["gs_ratio"] = df["gs_ratio"].pct_change()
    return chg.dropna()


def analyze_timeframe(
    df_lvl: pd.DataFrame,
    lags: list[int],
    fwd_horizons: list[int],
    roll_win: int,
    unit: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    u = unit[0].lower()
    df_chg = compute_changes(df_lvl)

    # 1. Contemporaneous summary (Level + Return/Change + Rolling stability)
    contemp_rows = []
    roll_corrs = {}
    for col, (label, mode) in MACRO_COLS.items():
        r_lvl, p_lvl = stats.pearsonr(df_lvl["gold"], df_lvl[col])
        rho_lvl, _ = stats.spearmanr(df_lvl["gold"], df_lvl[col])
        r_chg, p_chg = stats.pearsonr(df_chg["gold"], df_chg[col])
        rho_chg, _ = stats.spearmanr(df_chg["gold"], df_chg[col])

        rc = df_chg["gold"].rolling(roll_win).corr(df_chg[col]).dropna()
        roll_corrs[col] = rc

        # Granger causality (does col Granger-cause gold return?)
        max_g_lag = 5 if u == "d" else 4
        with contextlib.redirect_stdout(io.StringIO()):
            gc_res = grangercausalitytests(df_chg[["gold", col]], maxlag=max_g_lag)
        best_g_lag, best_g_p = min(
            ((lag, res[0]["ssr_ftest"][1]) for lag, res in gc_res.items()),
            key=lambda x: x[1],
        )

        contemp_rows.append(
            {
                "timeframe": unit,
                "macro": col,
                "label": label,
                "level_pearson_r": round(float(r_lvl), 4),
                "level_spearman_rho": round(float(rho_lvl), 4),
                "ret_pearson_r": round(float(r_chg), 4),
                "ret_p_value": round(float(p_chg), 6),
                "ret_spearman_rho": round(float(rho_chg), 4),
                f"roll_{roll_win}{u}_min": round(float(rc.min()), 3),
                f"roll_{roll_win}{u}_median": round(float(rc.median()), 3),
                f"roll_{roll_win}{u}_max": round(float(rc.max()), 3),
                "granger_best_lag": best_g_lag,
                "granger_min_p": round(float(best_g_p), 4),
            }
        )
    contemp_df = pd.DataFrame(contemp_rows)

    # 2. Lead-Lag cross-correlations (Macro change at t-k vs Gold return at t)
    leadlag_rows = []
    for col, (label, _) in MACRO_COLS.items():
        for k in lags:
            valid = pd.concat([df_chg["gold"], df_chg[col].shift(k)], axis=1).dropna()
            r_val, p_val = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
            rho_val, _ = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
            leadlag_rows.append(
                {
                    "timeframe": unit,
                    "macro": col,
                    "feature": f"{col}_chg_lag{k}{u}",
                    "lag": k,
                    "pearson_r": round(float(r_val), 4),
                    "p_value": round(float(p_val), 4),
                    "spearman_rho": round(float(rho_val), 4),
                    "sig_5pct": bool(p_val < 0.05),
                }
            )
    leadlag_df = pd.DataFrame(leadlag_rows)

    # 3. Macro trend/level predictors (known at t-1) vs multi-horizon forward Gold returns
    fwd_rets = {h: (df_lvl["gold"].shift(-(h - 1)) / df_lvl["gold"].shift(1)) - 1.0 for h in fwd_horizons}
    trend_rows = []
    ema_span = 20 if u == "d" else 8

    # Macro EMA spreads & momentum + Gold/Silver ratio level and z-score
    candidate_features: dict[str, pd.Series] = {}
    for col in MACRO_COLS:
        ema_prev = df_lvl[col].ewm(span=ema_span, adjust=False).mean().shift(1)
        ema_prev.iloc[:ema_span] = np.nan
        candidate_features[f"{col}_vs_ema{ema_span}{u}"] = (df_lvl[col].shift(1) / ema_prev) - 1.0
        # Trailing momentum over ema_span periods (known at t-1)
        candidate_features[f"{col}_mom_{ema_span}{u}"] = (
            df_lvl[col].diff(ema_span).shift(1)
            if col == "us10y"
            else df_lvl[col].pct_change(ema_span).shift(1)
        )
    candidate_features["gold_silver_ratio_lag1"] = df_lvl["gs_ratio"].shift(1)
    gs_ema = df_lvl["gs_ratio"].ewm(span=ema_span, adjust=False).mean().shift(1)
    gs_ema.iloc[:ema_span] = np.nan
    candidate_features[f"gs_ratio_vs_ema{ema_span}{u}"] = (df_lvl["gs_ratio"].shift(1) / gs_ema) - 1.0

    for fname, fseries in candidate_features.items():
        rec: dict = {"timeframe": unit, "feature": fname}
        for h in fwd_horizons:
            valid = pd.concat([fwd_rets[h], fseries], axis=1).dropna()
            r_val, p_val = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
            rec[f"r_fwd_{h}{u}"] = round(float(r_val), 4)
            rec[f"p_fwd_{h}{u}"] = round(float(p_val), 4)
        trend_rows.append(rec)
    trend_df = pd.DataFrame(trend_rows)

    return contemp_df, leadlag_df, trend_df, roll_corrs


def plot_dashboard(
    d_contemp: pd.DataFrame,
    w_contemp: pd.DataFrame,
    d_roll: dict[str, pd.Series],
    w_roll: dict[str, pd.Series],
    d_trend: pd.DataFrame,
    w_trend: pd.DataFrame,
) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Macro Predictors vs. Gold (GC=F): Daily & Weekly Correlation Analysis (2021–2026)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Contemporaneous Return Correlations (Daily vs Weekly)
    ax = axes[0, 0]
    labels = [MACRO_COLS[c][0] for c in MACRO_COLS]
    x = np.arange(len(labels))
    width = 0.35
    d_vals = [d_contemp.loc[d_contemp["macro"] == c, "ret_pearson_r"].iloc[0] for c in MACRO_COLS]
    w_vals = [w_contemp.loc[w_contemp["macro"] == c, "ret_pearson_r"].iloc[0] for c in MACRO_COLS]
    b1 = ax.bar(x - width / 2, d_vals, width, label="Daily Returns", color="#2b6cb0")
    b2 = ax.bar(x + width / 2, w_vals, width, label="Weekly Returns", color="#dd6b20")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(-0.7, 0.95)
    ax.set_ylabel("Pearson r")
    ax.set_title("Same-Period Return/Change Correlation with Gold")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 0.02 if h >= 0 else -0.02
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:+.2f}", ha="center", va=va, fontsize=8)

    # Panel 2: Rolling 60-Day Daily Return Correlations Over Time
    ax = axes[0, 1]
    palette = {"silver": "#718096", "dxy": "#e53e3e", "us10y": "#3182ce", "oil": "#38a169"}
    for col, (label, _) in MACRO_COLS.items():
        ax.plot(d_roll[col].index, d_roll[col].values, label=label, color=palette[col], linewidth=1.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("60-Trading-Day Rolling Daily Return Correlation with Gold")
    ax.set_ylabel("Rolling Pearson r")
    ax.set_ylim(-0.85, 0.98)
    ax.legend(loc="lower left", fontsize=8, ncol=2)

    # Panel 3: Daily Macro Trend/Ratio Features vs Forward Gold Returns Heatmap
    ax = axes[1, 0]
    d_mat = d_trend.set_index("feature")[["r_fwd_1d", "r_fwd_5d", "r_fwd_21d", "r_fwd_63d"]]
    d_mat.columns = ["1-Day", "5-Day", "21-Day", "63-Day"]
    sns.heatmap(
        d_mat,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-0.45,
        vmax=0.45,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
    )
    ax.set_title("Daily Lagged Macro Features (t-1) vs. Forward Gold Returns")
    ax.set_ylabel("")

    # Panel 4: Weekly Macro Trend/Ratio Features vs Forward Gold Returns Heatmap
    ax = axes[1, 1]
    w_mat = w_trend.set_index("feature")[["r_fwd_1w", "r_fwd_4w", "r_fwd_12w", "r_fwd_26w"]]
    w_mat.columns = ["1-Week", "4-Week", "12-Week", "26-Week"]
    sns.heatmap(
        w_mat,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-0.55,
        vmax=0.55,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
    )
    ax.set_title("Weekly Lagged Macro Features (t-1w) vs. Forward Gold Returns")
    ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "macro_correlation_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    gold = pd.read_csv(GOLD_CSV, parse_dates=["date"]).set_index("date").sort_index()[["close"]].rename(columns={"close": "gold"})
    macro = pd.read_csv(MACRO_CSV, parse_dates=["date"]).set_index("date").sort_index()

    daily_lvl = gold.join(macro, how="left")
    for col in MACRO_COLS:
        daily_lvl[col] = daily_lvl[col].ffill().bfill()
    daily_lvl["gs_ratio"] = daily_lvl["gold"] / daily_lvl["silver"]

    weekly_lvl = daily_lvl.resample("W-FRI").last().dropna()

    d_contemp, d_leadlag, d_trend, d_roll = analyze_timeframe(
        daily_lvl, lags=[1, 2, 3, 5, 10, 21], fwd_horizons=[1, 5, 21, 63], roll_win=60, unit="Days"
    )
    w_contemp, w_leadlag, w_trend, w_roll = analyze_timeframe(
        weekly_lvl, lags=[1, 2, 3, 4, 6, 8, 12], fwd_horizons=[1, 4, 12, 26], roll_win=26, unit="Weeks"
    )

    d_contemp.to_csv(OUT_DIR / "macro_daily_contemp.csv", index=False)
    w_contemp.to_csv(OUT_DIR / "macro_weekly_contemp.csv", index=False)
    d_leadlag.to_csv(OUT_DIR / "macro_daily_leadlag.csv", index=False)
    w_leadlag.to_csv(OUT_DIR / "macro_weekly_leadlag.csv", index=False)
    d_trend.to_csv(OUT_DIR / "macro_daily_forward.csv", index=False)
    w_trend.to_csv(OUT_DIR / "macro_weekly_forward.csv", index=False)

    chart_path = plot_dashboard(d_contemp, w_contemp, d_roll, w_roll, d_trend, w_trend)

    print("=== 1. CONTEMPORANEOUS CORRELATIONS & GRANGER CAUSALITY (DAILY) ===")
    print(d_contemp.to_string(index=False))

    print("\n=== 2. CONTEMPORANEOUS CORRELATIONS & GRANGER CAUSALITY (WEEKLY) ===")
    print(w_contemp.to_string(index=False))

    print("\n=== 3. TOP DAILY LEAD-LAG CORRELATIONS (Macro t-k -> Gold Return t) ===")
    print(d_leadlag.reindex(d_leadlag["pearson_r"].abs().sort_values(ascending=False).index).head(10).to_string(index=False))

    print("\n=== 4. TOP WEEKLY LEAD-LAG CORRELATIONS (Macro t-k -> Gold Return t) ===")
    print(w_leadlag.reindex(w_leadlag["pearson_r"].abs().sort_values(ascending=False).index).head(10).to_string(index=False))

    print("\n=== 5. DAILY LAGGED MACRO TREND/RATIO vs FORWARD GOLD RETURNS ===")
    print(d_trend[["feature", "r_fwd_1d", "r_fwd_5d", "r_fwd_21d", "r_fwd_63d", "p_fwd_63d"]].to_string(index=False))

    print("\n=== 6. WEEKLY LAGGED MACRO TREND/RATIO vs FORWARD GOLD RETURNS ===")
    print(w_trend[["feature", "r_fwd_1w", "r_fwd_4w", "r_fwd_12w", "r_fwd_26w", "p_fwd_26w"]].to_string(index=False))

    print(f"\nSaved Chart: {chart_path}")


if __name__ == "__main__":
    main()
