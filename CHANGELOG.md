# Changelog

All notable changes to the **GoldSniper** repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-11-22

### Added
- **Auto-Polling Live Intraday Quote Strip (`app.py`)**:
  - Added `fetch_live_gold_quote()` and `_render_live_quote_body()` inside a Streamlit `st.fragment(run_every=...)` loop with configurable intervals (`Off`, `15 sec`, `30 sec`, `60 sec`) and manual poll trigger, displaying current `GC=F` session price, intraday high-low range, contract volume, and real-time spread to the model target without retraining settled daily models.
- **Incremental Delta Ingestion & Concurrent Market Sync (`fetch_gold_5y_daily.py`, `fetch_macro_5y_daily.py`, `run_pipeline.py`)**:
  - Upgraded Gold and 12-series macro ingestion to fetch bounded `period1`/`period2` deltas with a 7-day settlement overlap window, concurrent 6-worker macro downloads, and `--full` flag overrides.
  - Added in-process `sync_market_data()` in `app.py` with a `Force Full 5-Year Refetch` option, cutting market refresh time from multi-second full pulls to under one second on incremental runs.
- **Expanded Verification Suite (`tests/test_goldsniper.py`)**:
  - Added `test_forecast_figure_tooltip_and_default_3m`, `test_incremental_gold_merge`, and `test_incremental_macro_merge`.

### Changed
- **Default 3M Chart Window & Tooltip Clarity (`app.py`)**:
  - Extracted `build_forecast_figure()` and set the initial x-axis window to `3M` when `Full 5-Year History` is unchecked without restricting y-axis rescaling on `6M`, `1Y`, or `All`.
  - Separated the green dashed anchor connector (`hoverinfo="skip"`) from the `Live Target` star marker so hovering on the latest settled close no longer duplicates `Actual Close`.
- **Refreshed Market Datasets (`data/`)**:
  - Synchronized Gold Futures (`GC=F`), 12 macroeconomic series, cleaned trading/business-day datasets, and daily/weekly feature matrices through `2026-11-20`.

---

## [1.0.0] - 2026-11-19

### Added
- **Publish-Ready GoldSniper Streamlit Storytelling Application (`app.py`)**:
  - Designed a 4-chapter narrative flow (`Live Forecast Story`, `Proof of 90% Accuracy`, `Why Macro Moves Gold`, and `Model Arena`).
  - Added executive takeaway banners that dynamically interpret the active forecast horizon, projected price move, confidence interval, and out-of-sample skill score.
  - Added contextual native tooltips (`help=...`) across all KPI metrics, sidebar controls, and dataframe columns (`st.column_config`) while enforcing a clean, zero-clutter visual layout without em dashes or parentheses in UI strings.
  - Added one-click CSV export of walk-forward out-of-sample forecast ledgers and live market refresh from Yahoo Finance.
- **>90% Accuracy Walk-Forward Optimization Engine (`optimize_90_accuracy.py`)**:
  - Engineered 70 Daily and 67 Weekly strictly lagged (`t-1`) predictors across 12 macroeconomic and cross-asset series.
  - Discovered and validated the **COMEX (1:30 PM ET) vs. NYSE (4:00 PM ET) `Gold/GDX` cross-market lead-lag signal** (`r = -0.1641, t = -5.89, p < 10^{-8}`), enabling daily 1-day models (`HistGBR`, `RandomForest`, `Ridge_Strong`, `Ensemble_Stack`) to beat Naive Persistence out of sample.
  - Implemented volatility-scaled expanding conformal prediction intervals achieving `>90%` empirical coverage on Daily 1-Day (`91.07%`) and Weekly 1-Week (`90.24%`).
  - Implemented causal moving-average trend slope (`SMA_20d: 94.64%`, `SMA_12w: 95.12%`) and regime state (`EMA_12w: 90.24%`) walk-forward classifiers.
- **6-Tier Multi-Algorithm Forecasting Benchmark (`forecast_benchmark.py`)**:
  - Evaluated Naive Persistence, Naive + Drift, `ARIMA(1,1,0)`, `ARIMAX(1,0,0)`, Holt's Damped Trend ETS, `Ridge`, `ElasticNet`, `RandomForest`, and `HistGradientBoosting` across 7 feature mix-match bundles.
- **Exploratory Correlation & Causality Suites**:
  - `explore_predictors.py`: Short-window lag, SMA, EMA, and PACF analysis on Daily and Weekly timeframes.
  - `explore_bigger_ma_ema.py`: Medium-to-long-window (`30d–500d` and `12w–104w`) SMA/EMA spreads and crossover ratios vs. multi-horizon forward returns.
  - `explore_macro_correlations.py`: Contemporaneous, lead-lag, rolling 60-day/26-week correlation stability, and Granger causality tests.
- **Keyless 5-Year Data Ingestion & Cleaning Pipeline**:
  - `fetch_gold_5y_daily.py`: Automated 5-year daily OHLCV ingestion for COMEX Gold Futures (`GC=F`).
  - `fetch_macro_5y_daily.py`: Automated 5-year daily ingestion for 12 macroeconomic and cross-asset tickers (`DX-Y.NYB`, `^TNX`, `^FVX`, `^TYX`, `SI=F`, `CL=F`, `HG=F`, `^VIX`, `^GSPC`, `GDX`, `EURUSD=X`, `JPY=X`).
  - `clean_gold_data.py`: Removes unsettled intraday bars, rounds IEEE-754 float noise, imputes zero-volume anomalies via 21-day centered rolling median, and exports both trading-day (`gold_daily_5y_clean.csv`) and uniform business-day (`gold_daily_5y_bday.csv`) tables.
  - `build_gold_features.py`: Builds leak-free daily and weekly feature matrices.
- **Repository Packaging, Documentation & Automated Tests**:
  - Added `run_pipeline.py`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `.streamlit/config.toml`, `LICENSE`, `CONTRIBUTING.md`, comprehensive `docs/` guides, and `tests/test_goldsniper.py`.
