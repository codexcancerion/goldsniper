"""
Gold (GC=F) Multi-Algorithm Forecasting Benchmark & Feature Mix-Match Experiment.

Evaluates 6 algorithm tiers across 4 horizons using strict Walk-Forward (Expanding-Window)
validation on 2025-05-01 -> 2026-11-18 (covering both the rally to $5,318 and 2026 pullback):
  - Daily  : 1-day ahead (h=1d), 5-day ahead (h=5d)
  - Weekly : 1-week ahead (h=1w), 4-week ahead (h=4w)

Algorithms:
  0. Naive Persistence & Naive + Drift
  1. ARIMA(1,1,0) & ARIMAX(1,1,0) [statsmodels]
  2. Holt's Damped Trend ETS      [statsmodels]
  3. Ridge & ElasticNet           [scikit-learn]
  4. Random Forest & HistGradientBoosting [scikit-learn]

Feature Mix-Match Sets (all strictly t-1 shifted, leak-free):
  - Gold_Short
  - Gold_BigMA
  - Macro_Core
  - GoldShort_plus_Macro
  - Top_Correlators
  - Top_Corr_plus_BigMA
  - All_Combined

Outputs:
  - outputs/forecast_experiment_results.csv
  - outputs/forecast_best_models.png
  - outputs/forecast_feature_mixmatch_heatmap.png
"""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
GOLD_CSV = BASE_DIR / "data" / "gold_daily_5y_clean.csv"
MACRO_CSV = BASE_DIR / "data" / "macro_daily_5y.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = pd.Timestamp("2025-05-01")


def build_rich_daily_table() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    gold = pd.read_csv(GOLD_CSV, parse_dates=["date"]).set_index("date").sort_index()
    macro = pd.read_csv(MACRO_CSV, parse_dates=["date"]).set_index("date").sort_index()

    df = gold[["close", "high", "low", "volume"]].join(macro, how="left")
    for col in ["dxy", "us10y", "silver", "oil"]:
        df[col] = df[col].ffill().bfill()

    c = df["close"]
    ret = c.pct_change()
    feat = pd.DataFrame(index=df.index)
    feat["close"] = c

    # Short-window Gold features (t-1 shifted)
    sma_3 = c.rolling(3).mean().shift(1)
    ema_5 = c.ewm(span=5, adjust=False).mean().shift(1)
    ema_10 = c.ewm(span=10, adjust=False).mean().shift(1)
    feat["sma_3d_spread"] = (c.shift(1) / sma_3) - 1.0
    feat["ema_5d_vs_10d"] = (ema_5 / ema_10) - 1.0
    feat["lag_1d_return"] = ret.shift(1)
    feat["lag_7d_return"] = ret.shift(7)
    feat["vol_5d"] = ret.rolling(5).std().shift(1)
    feat["vol_20d"] = ret.rolling(20).std().shift(1)
    feat["hl_range_lag1"] = ((df["high"] - df["low"]) / c).shift(1)

    # Medium & Big MA / EMA Gold regime features (t-1 shifted; use min_periods for early burn-in)
    sma_20 = c.rolling(20).mean().shift(1)
    sma_100 = c.rolling(100).mean().shift(1)
    ema_200 = c.ewm(span=200, adjust=False).mean().shift(1)
    ema_500 = c.ewm(span=500, adjust=False).mean().shift(1)
    feat["sma_20d_vs_100d"] = (sma_20 / sma_100) - 1.0
    feat["ema_200d_vs_500d"] = (ema_200 / ema_500) - 1.0
    feat["close_vs_ema200d"] = (c.shift(1) / ema_200) - 1.0
    feat["close_vs_ema500d"] = (c.shift(1) / ema_500) - 1.0

    # Macro features (t-1 shifted)
    feat["dxy_ret_lag1"] = df["dxy"].pct_change().shift(1)
    feat["us10y_chg_lag1"] = df["us10y"].diff().shift(1)
    feat["silver_ret_lag1"] = df["silver"].pct_change().shift(1)
    feat["oil_ret_lag1"] = df["oil"].pct_change().shift(1)

    feat["dxy_vs_ema20d"] = (df["dxy"].shift(1) / df["dxy"].ewm(span=20, adjust=False).mean().shift(1)) - 1.0
    feat["us10y_vs_ema20d"] = (df["us10y"].shift(1) / df["us10y"].ewm(span=20, adjust=False).mean().shift(1)) - 1.0
    feat["oil_vs_ema20d"] = (df["oil"].shift(1) / df["oil"].ewm(span=20, adjust=False).mean().shift(1)) - 1.0
    feat["oil_mom_20d"] = df["oil"].pct_change(20).shift(1)
    feat["oil_lag1"] = df["oil"].shift(1)
    feat["gold_silver_ratio_lag1"] = (c.shift(1) / df["silver"].shift(1))
    gs_ema20 = (c / df["silver"]).ewm(span=20, adjust=False).mean().shift(1)
    feat["gs_ratio_vs_ema20d"] = (feat["gold_silver_ratio_lag1"] / gs_ema20) - 1.0

    # Keep from row 105 onward (after sma_100 burn-in; ema_500 is already well-formed)
    feat = feat.iloc[105:].dropna().copy()

    feature_sets = {
        "Gold_Short": ["sma_3d_spread", "ema_5d_vs_10d", "lag_1d_return", "lag_7d_return", "vol_5d", "vol_20d", "hl_range_lag1"],
        "Gold_BigMA": ["sma_20d_vs_100d", "ema_200d_vs_500d", "close_vs_ema200d", "close_vs_ema500d"],
        "Macro_Core": ["dxy_ret_lag1", "us10y_chg_lag1", "silver_ret_lag1", "oil_ret_lag1", "dxy_vs_ema20d", "us10y_vs_ema20d", "oil_mom_20d"],
        "GoldShort_plus_Macro": [
            "sma_3d_spread", "ema_5d_vs_10d", "lag_1d_return", "vol_20d",
            "dxy_vs_ema20d", "us10y_vs_ema20d", "oil_mom_20d", "silver_ret_lag1",
        ],
        "Top_Correlators": [
            "oil_lag1", "oil_mom_20d", "gold_silver_ratio_lag1", "dxy_vs_ema20d",
            "us10y_vs_ema20d", "sma_3d_spread",
        ],
        "Top_Corr_plus_BigMA": [
            "oil_lag1", "oil_mom_20d", "gold_silver_ratio_lag1", "dxy_vs_ema20d",
            "us10y_vs_ema20d", "sma_20d_vs_100d", "ema_200d_vs_500d",
        ],
        "All_Combined": [col for col in feat.columns if col != "close"],
    }
    return feat, feature_sets


def build_rich_weekly_table() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    gold = pd.read_csv(GOLD_CSV, parse_dates=["date"]).set_index("date").sort_index()
    macro = pd.read_csv(MACRO_CSV, parse_dates=["date"]).set_index("date").sort_index()

    daily = gold[["close", "high", "low"]].join(macro, how="left")
    for col in ["dxy", "us10y", "silver", "oil"]:
        daily[col] = daily[col].ffill().bfill()

    wk = pd.DataFrame(
        {
            "close": daily["close"].resample("W-FRI").last(),
            "high": daily["high"].resample("W-FRI").max(),
            "low": daily["low"].resample("W-FRI").min(),
            "dxy": daily["dxy"].resample("W-FRI").last(),
            "us10y": daily["us10y"].resample("W-FRI").last(),
            "silver": daily["silver"].resample("W-FRI").last(),
            "oil": daily["oil"].resample("W-FRI").last(),
        }
    ).dropna()

    c = wk["close"]
    ret = c.pct_change()
    feat = pd.DataFrame(index=wk.index)
    feat["close"] = c

    # Short-window Weekly Gold features (t-1w shifted)
    ema_2 = c.ewm(span=2, adjust=False).mean().shift(1)
    ema_4 = c.ewm(span=4, adjust=False).mean().shift(1)
    feat["ema_2w_vs_4w"] = (ema_2 / ema_4) - 1.0
    feat["lag_1w_return"] = ret.shift(1)
    feat["lag_4w_return"] = ret.shift(4)
    feat["lag_6w_return"] = ret.shift(6)
    feat["vol_4w"] = ret.rolling(4).std().shift(1)
    feat["hl_range_lag1w"] = ((wk["high"] - wk["low"]) / c).shift(1)

    # Medium & Big Weekly MA / EMA Gold regime features
    sma_20 = c.rolling(20).mean().shift(1)
    ema_26 = c.ewm(span=26, adjust=False).mean().shift(1)
    ema_104 = c.ewm(span=104, adjust=False).mean().shift(1)
    feat["close_vs_sma20w"] = (c.shift(1) / sma_20) - 1.0
    feat["ema_26w_vs_104w"] = (ema_26 / ema_104) - 1.0
    feat["close_vs_ema104w"] = (c.shift(1) / ema_104) - 1.0

    # Weekly Macro features (t-1w shifted)
    feat["dxy_ret_lag1w"] = wk["dxy"].pct_change().shift(1)
    feat["us10y_chg_lag1w"] = wk["us10y"].diff().shift(1)
    feat["silver_ret_lag1w"] = wk["silver"].pct_change().shift(1)
    feat["oil_ret_lag1w"] = wk["oil"].pct_change().shift(1)

    feat["dxy_vs_ema8w"] = (wk["dxy"].shift(1) / wk["dxy"].ewm(span=8, adjust=False).mean().shift(1)) - 1.0
    feat["us10y_vs_ema8w"] = (wk["us10y"].shift(1) / wk["us10y"].ewm(span=8, adjust=False).mean().shift(1)) - 1.0
    feat["oil_vs_ema8w"] = (wk["oil"].shift(1) / wk["oil"].ewm(span=8, adjust=False).mean().shift(1)) - 1.0
    feat["oil_mom_8w"] = wk["oil"].pct_change(8).shift(1)
    feat["oil_lag1w"] = wk["oil"].shift(1)
    feat["gold_silver_ratio_lag1w"] = (c.shift(1) / wk["silver"].shift(1))

    feat = feat.iloc[22:].dropna().copy()

    feature_sets = {
        "Gold_Short": ["ema_2w_vs_4w", "lag_1w_return", "lag_4w_return", "lag_6w_return", "vol_4w", "hl_range_lag1w"],
        "Gold_BigMA": ["close_vs_sma20w", "ema_26w_vs_104w", "close_vs_ema104w"],
        "Macro_Core": ["dxy_ret_lag1w", "us10y_chg_lag1w", "silver_ret_lag1w", "oil_ret_lag1w", "dxy_vs_ema8w", "us10y_vs_ema8w", "oil_mom_8w"],
        "GoldShort_plus_Macro": [
            "ema_2w_vs_4w", "lag_4w_return", "vol_4w",
            "dxy_vs_ema8w", "us10y_vs_ema8w", "oil_mom_8w", "silver_ret_lag1w",
        ],
        "Top_Correlators": [
            "oil_lag1w", "oil_mom_8w", "gold_silver_ratio_lag1w", "dxy_vs_ema8w",
            "us10y_vs_ema8w", "ema_2w_vs_4w",
        ],
        "Top_Corr_plus_BigMA": [
            "oil_lag1w", "oil_mom_8w", "gold_silver_ratio_lag1w", "dxy_vs_ema8w",
            "us10y_vs_ema8w", "close_vs_sma20w", "ema_26w_vs_104w",
        ],
        "All_Combined": [col for col in feat.columns if col != "close"],
    }
    return feat, feature_sets


def compute_metrics(
    actual_price: np.ndarray,
    pred_price: np.ndarray,
    anchor_price: np.ndarray,
    naive_rmse: float | None = None,
) -> dict:
    err = actual_price - pred_price
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / actual_price)) * 100.0)
    actual_dir = np.sign(actual_price - anchor_price)
    pred_dir = np.sign(pred_price - anchor_price)
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100.0)
    skill = float((1.0 - rmse / naive_rmse) * 100.0) if naive_rmse else 0.0
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape_pct": round(mape, 3),
        "dir_acc_pct": round(dir_acc, 2),
        "skill_vs_naive_pct": round(skill, 2),
    }


def evaluate_horizon(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    horizon: int,
    horizon_name: str,
    refit_every: int,
) -> tuple[list[dict], dict[str, pd.Series]]:
    """
    Strictly leak-free walk-forward evaluation for h-step ahead forecast.
    At row i (date t), features X[i] are known at close of t-1 (anchor = close[i-1]).
    Target is close[i + h - 1], so target log-return is log(close[i + h - 1] / close[i - 1]).
    When predicting at row i, training rows must satisfy j + h - 1 <= i - 1 => j <= i - h.
    """
    work = df.copy()
    work["anchor"] = work["close"].shift(1)
    work["target_price"] = work["close"].shift(-(horizon - 1))
    work["target_log_ret"] = np.log(work["target_price"] / work["anchor"])
    work["target_date"] = work.index.to_series().shift(-(horizon - 1))
    work = work.dropna(subset=["anchor", "target_price", "target_log_ret"]).copy()

    test_indices = np.where(work.index >= TEST_START)[0]
    actuals = work["target_price"].iloc[test_indices].values
    anchors = work["anchor"].iloc[test_indices].values
    target_dates = work["target_date"].iloc[test_indices].values

    results: list[dict] = []
    pred_store: dict[str, pd.Series] = {
        "Actual": pd.Series(actuals, index=target_dates),
        "Naive": pd.Series(anchors, index=target_dates),
    }

    # --- Tier 0a: Naive Persistence ---
    naive_m = compute_metrics(actuals, anchors, anchors)
    naive_rmse = naive_m["rmse"]
    results.append({"horizon": horizon_name, "model": "Naive_Persistence", "feature_set": "Lag_1_Only", "n_test": len(actuals), **naive_m})

    # --- Tier 0b: Naive + Expanding Drift ---
    drift_preds = []
    for i in test_indices:
        train_r = work["target_log_ret"].iloc[: i - horizon + 1].values
        mu = float(np.mean(train_r))
        drift_preds.append(work["anchor"].iloc[i] * np.exp(mu))
    drift_preds = np.array(drift_preds)
    m = compute_metrics(actuals, drift_preds, anchors, naive_rmse)
    results.append({"horizon": horizon_name, "model": "Naive_Drift", "feature_set": "LogReturn_Mean", "n_test": len(actuals), **m})
    pred_store["Naive_Drift"] = pd.Series(drift_preds, index=target_dates)

    # --- Tier 1: ARIMA(1,1,0) on log_close (expanding window, refit periodically) ---
    log_c_all = np.log(df["close"])
    work_dates = work.index
    arima_preds = []
    last_fit_i = -999
    ar1_phi = 0.0
    ar1_drift = 0.0
    for i in test_indices:
        anchor_date = df.index[df.index.get_loc(work_dates[i]) - 1]
        if i - last_fit_i >= refit_every:
            hist = log_c_all.loc[:anchor_date].values
            try:
                fit = ARIMA(hist, order=(1, 1, 0), trend="t").fit()
                fc = fit.forecast(steps=horizon)[-1]
                # Store simple AR(1) params for intermediate steps
                ar1_phi = float(fit.arparams[0]) if len(fit.arparams) else 0.0
                ar1_drift = float(np.mean(np.diff(hist)))
                last_fit_i = i
                arima_preds.append(float(np.exp(fc)))
                continue
            except Exception:
                pass
        # Fast multi-step AR(1)+drift recursion between refits
        hist = log_c_all.loc[:anchor_date].values
        last_diff = hist[-1] - hist[-2]
        cur_log = hist[-1]
        d = last_diff
        for _ in range(horizon):
            d = ar1_drift + ar1_phi * (d - ar1_drift)
            cur_log += d
        arima_preds.append(float(np.exp(cur_log)))
    arima_preds = np.array(arima_preds)
    m = compute_metrics(actuals, arima_preds, anchors, naive_rmse)
    results.append({"horizon": horizon_name, "model": "ARIMA_110", "feature_set": "Univariate_LogClose", "n_test": len(actuals), **m})
    pred_store["ARIMA_110"] = pd.Series(arima_preds, index=target_dates)

    # --- Tier 2: Holt's Damped Trend ETS on log_close ---
    ets_preds = []
    last_ets_fit = None
    last_ets_i = -999
    for i in test_indices:
        anchor_date = df.index[df.index.get_loc(work_dates[i]) - 1]
        hist = log_c_all.loc[:anchor_date].values
        if i - last_ets_i >= refit_every or last_ets_fit is None:
            try:
                model = ExponentialSmoothing(hist, trend="add", damped_trend=True, initialization_method="estimated")
                last_ets_fit = model.fit(optimized=True)
                last_ets_i = i
                fc = float(last_ets_fit.forecast(horizon)[-1])
                ets_preds.append(float(np.exp(fc)))
                continue
            except Exception:
                pass
        # Between refits, update quick damped trend using fitted params
        alpha = float(last_ets_fit.params.get("smoothing_level", 0.8))
        beta = float(last_ets_fit.params.get("smoothing_trend", 0.05))
        phi = float(last_ets_fit.params.get("damping_trend", 0.9))
        lvl = hist[-2]
        tr = hist[-2] - hist[-3]
        new_lvl = alpha * hist[-1] + (1 - alpha) * (lvl + phi * tr)
        new_tr = beta * (new_lvl - lvl) + (1 - beta) * phi * tr
        phi_sum = sum(phi ** k for k in range(1, horizon + 1))
        ets_preds.append(float(np.exp(new_lvl + phi_sum * new_tr)))
    ets_preds = np.array(ets_preds)
    m = compute_metrics(actuals, ets_preds, anchors, naive_rmse)
    results.append({"horizon": horizon_name, "model": "Holt_Damped_ETS", "feature_set": "Univariate_LogClose", "n_test": len(actuals), **m})
    pred_store["Holt_Damped_ETS"] = pd.Series(ets_preds, index=target_dates)

    # --- Tier 1b: ARIMAX (ARIMA(1,1,0) + Top_Correlators exogenous on log_return) ---
    top_cols = feature_sets["Top_Correlators"]
    arimax_preds = []
    last_fit_i = -999
    fitted_arimax = None
    for i in test_indices:
        train_end = i - horizon + 1
        X_tr = work[top_cols].iloc[:train_end].values
        y_tr = work["target_log_ret"].iloc[:train_end].values
        X_te = work[top_cols].iloc[i : i + 1].values
        if i - last_fit_i >= refit_every or fitted_arimax is None:
            # Standardize exog inside expanding train window
            mu_x = X_tr.mean(axis=0)
            sd_x = np.where(X_tr.std(axis=0) == 0, 1.0, X_tr.std(axis=0))
            X_tr_s = (X_tr - mu_x) / sd_x
            try:
                fitted_arimax = (ARIMA(y_tr, exog=X_tr_s, order=(1, 0, 0)).fit(), mu_x, sd_x)
                last_fit_i = i
            except Exception:
                fitted_arimax = None
        if fitted_arimax is not None:
            fit_obj, mu_x, sd_x = fitted_arimax
            X_te_s = (X_te - mu_x) / sd_x
            # Use linear exogenous + intercept prediction cleanly
            params = fit_obj.params
            const = params[0]
            betas = params[1 : 1 + len(top_cols)]
            pred_r = float(const + np.dot(X_te_s[0], betas))
            pred_r = float(np.clip(pred_r, -0.25, 0.25))
            arimax_preds.append(work["anchor"].iloc[i] * np.exp(pred_r))
        else:
            arimax_preds.append(work["anchor"].iloc[i])
    arimax_preds = np.array(arimax_preds)
    m = compute_metrics(actuals, arimax_preds, anchors, naive_rmse)
    results.append({"horizon": horizon_name, "model": "ARIMAX_100_Ret", "feature_set": "Top_Correlators", "n_test": len(actuals), **m})
    pred_store["ARIMAX_TopCorr"] = pd.Series(arimax_preds, index=target_dates)

    # --- Tiers 3 & 4: ML Models × 7 Feature Mix-Match Sets ---
    model_factories = {
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=50.0)),
        "ElasticNet": lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.002, l1_ratio=0.5, max_iter=2000)),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=120, max_depth=4, min_samples_leaf=15, random_state=42, n_jobs=-1
        ),
        "HistGBR": lambda: HistGradientBoostingRegressor(
            max_iter=80, max_depth=3, learning_rate=0.05, min_samples_leaf=15, l2_regularization=5.0, random_state=42
        ),
    }

    best_ml_rmse = float("inf")
    best_ml_key = ""
    best_ml_series = None

    for mname, factory in model_factories.items():
        step_refit = 5 if mname in ("Ridge", "ElasticNet") else refit_every
        for fset_name, cols in feature_sets.items():
            X_all = work[cols].values
            y_all = work["target_log_ret"].values
            preds = np.empty(len(test_indices), dtype=float)
            model = None
            last_fit = -999

            for idx_pos, i in enumerate(test_indices):
                if i - last_fit >= step_refit or model is None:
                    train_end = i - horizon + 1
                    model = factory()
                    model.fit(X_all[:train_end], y_all[:train_end])
                    last_fit = i
                pred_r = float(model.predict(X_all[i : i + 1])[0])
                pred_r = float(np.clip(pred_r, -0.30, 0.30))
                preds[idx_pos] = work["anchor"].iloc[i] * np.exp(pred_r)

            m = compute_metrics(actuals, preds, anchors, naive_rmse)
            results.append(
                {
                    "horizon": horizon_name,
                    "model": mname,
                    "feature_set": fset_name,
                    "n_test": len(actuals),
                    **m,
                }
            )
            if m["rmse"] < best_ml_rmse:
                best_ml_rmse = m["rmse"]
                best_ml_key = f"{mname} ({fset_name})"
                best_ml_series = pd.Series(preds, index=target_dates)

    if best_ml_series is not None:
        pred_store[f"Best_ML: {best_ml_key}"] = best_ml_series

    return results, pred_store


def plot_results(res_df: pd.DataFrame, all_preds: dict[str, dict[str, pd.Series]]) -> tuple[Path, Path]:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Best Model Forecast vs Actual across all 4 horizons
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Gold (GC=F) Walk-Forward Out-of-Sample Forecasts (May 2025 – Nov 2026)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    horizons = ["Daily_1d", "Daily_5d", "Weekly_1w", "Weekly_4w"]
    for ax, hname in zip(axes.flat, horizons):
        pdict = all_preds[hname]
        best_row = res_df[res_df["horizon"] == hname].sort_values("rmse").iloc[0]
        naive_row = res_df[(res_df["horizon"] == hname) & (res_df["model"] == "Naive_Persistence")].iloc[0]

        ax.plot(pdict["Actual"].index, pdict["Actual"].values, label="Actual Close", color="#1a202c", linewidth=2.0)
        ax.plot(pdict["Naive"].index, pdict["Naive"].values, label=f"Naive (RMSE ${naive_row['rmse']:.1f})", color="#a0aec0", linestyle="--", linewidth=1.3)

        # Plot best ML or best overall
        ml_keys = [k for k in pdict if k.startswith("Best_ML:")]
        if ml_keys:
            k = ml_keys[0]
            ml_sub = res_df[(res_df["horizon"] == hname) & (res_df["model"].isin(["Ridge", "ElasticNet", "RandomForest", "HistGBR"]))].sort_values("rmse").iloc[0]
            ax.plot(
                pdict[k].index,
                pdict[k].values,
                label=f"{k.replace('Best_ML: ', '')} (RMSE ${ml_sub['rmse']:.1f})",
                color="#e53e3e",
                linewidth=1.6,
            )
        if best_row["model"] in pdict:
            ax.plot(
                pdict[best_row["model"]].index,
                pdict[best_row["model"]].values,
                label=f"Winner: {best_row['model']} (RMSE ${best_row['rmse']:.1f})",
                color="#2b6cb0",
                linewidth=1.5,
                alpha=0.85,
            )

        ax.set_title(f"{hname} | Best: {best_row['model']} [{best_row['feature_set']}] (Skill {best_row['skill_vs_naive_pct']:+.2f}%)")
        ax.set_ylabel("Gold Price (USD)")
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fc_path = OUT_DIR / "forecast_best_models.png"
    fig.savefig(fc_path, dpi=150)
    plt.close(fig)

    # 2. Feature Mix-Match Heatmap (Skill Score vs Naive across Models × Feature Sets)
    ml_df = res_df[res_df["model"].isin(["Ridge", "ElasticNet", "RandomForest", "HistGBR"])].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Feature Mix-Match Experiment: Skill Score vs. Naive (%) — Positive = Beats Naive",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    for ax, hname in zip(axes.flat, horizons):
        sub = ml_df[ml_df["horizon"] == hname]
        piv = sub.pivot(index="feature_set", columns="model", values="skill_vs_naive_pct")
        piv = piv[["Ridge", "ElasticNet", "RandomForest", "HistGBR"]]
        sns.heatmap(
            piv,
            ax=ax,
            cmap="RdYlGn",
            center=0,
            annot=True,
            fmt="+.2f",
            linewidths=0.5,
            cbar_kws={"label": "Skill vs Naive (%)"},
        )
        ax.set_title(f"{hname} Horizon")
        ax.set_ylabel("")
        ax.set_xlabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hm_path = OUT_DIR / "forecast_feature_mixmatch_heatmap.png"
    fig.savefig(hm_path, dpi=150)
    plt.close(fig)

    return fc_path, hm_path


def main() -> None:
    daily_df, daily_sets = build_rich_daily_table()
    weekly_df, weekly_sets = build_rich_weekly_table()

    all_results: list[dict] = []
    all_preds: dict[str, dict[str, pd.Series]] = {}

    specs = [
        ("Daily_1d", daily_df, daily_sets, 1, 15),
        ("Daily_5d", daily_df, daily_sets, 5, 15),
        ("Weekly_1w", weekly_df, weekly_sets, 1, 4),
        ("Weekly_4w", weekly_df, weekly_sets, 4, 4),
    ]

    for hname, data_df, fsets, h, refit_step in specs:
        print(f"Running walk-forward experiment for {hname} (h={h})...")
        res, preds = evaluate_horizon(data_df, fsets, h, hname, refit_every=refit_step)
        all_results.extend(res)
        all_preds[hname] = preds

    res_df = pd.DataFrame(all_results)
    csv_path = OUT_DIR / "forecast_experiment_results.csv"
    res_df.to_csv(csv_path, index=False)

    fc_path, hm_path = plot_results(res_df, all_preds)

    for hname in ["Daily_1d", "Daily_5d", "Weekly_1w", "Weekly_4w"]:
        print(f"\n=== LEADERBOARD: {hname} (Top 8 by Lowest RMSE) ===")
        sub = res_df[res_df["horizon"] == hname].sort_values("rmse").head(8)
        print(
            sub[["model", "feature_set", "rmse", "mae", "mape_pct", "dir_acc_pct", "skill_vs_naive_pct"]].to_string(
                index=False
            )
        )

    print(f"\nSaved Full Leaderboard : {csv_path}")
    print(f"Saved Forecast Chart   : {fc_path}")
    print(f"Saved Mix-Match Heatmap: {hm_path}")


if __name__ == "__main__":
    main()
