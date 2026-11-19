# GoldSniper Data Dictionary

This document provides the complete schema for every dataset in `data/` and defines all **12 macroeconomic tickers**, **70 Daily features**, and **67 Weekly features** engineered in **GoldSniper**.

---

## 1. Primary Datasets (`data/`)

| File Path | Frequency | Rows × Cols | Date Range | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `data/gold_daily_5y.csv` | Trading Days (`D`) | `1,258 × 7` | `2021-11-19` to `2026-11-19` | Raw uncleaned COMEX Gold Futures (`GC=F`) pull from Yahoo Finance. |
| `data/gold_daily_5y_clean.csv` | Settled Trading Days | `1,257 × 10` | `2021-11-19` to `2026-11-18` | Cleaned Gold OHLCV with `volume_imputed`, `log_close`, `daily_return`, and `log_return`. |
| `data/gold_daily_5y_bday.csv` | Uniform Business Days (`B`) | `1,304 × 11` | `2021-11-19` to `2026-11-18` | Uniform weekday index (`47` market holidays forward-filled with `is_holiday_filled` flag) for ARIMA/ETS. |
| `data/macro_daily_5y.csv` | Settled Trading Days | `1,260 × 13` | `2021-11-19` to `2026-11-18` | 12 daily macroeconomic and cross-asset closing series aligned by date. |
| `data/gold_daily_features.csv` | Settled Trading Days | `1,232 × 70` | `2021-12-28` to `2026-11-18` | Leak-free (`t-1` shifted) daily feature matrix after 25-day burn-in window. |
| `data/gold_weekly_features.csv` | Weekly Friday (`W-FRI`) | `249 × 67` | `2022-02-18` to `2026-11-20` | Leak-free (`t-1w` shifted) weekly feature matrix after 13-week burn-in window. |

---

## 2. External Macroeconomic & Cross-Asset Universe (`data/macro_daily_5y.csv`)

All 12 series are fetched without API keys via `fetch_macro_5y_daily.py`:

| Column | Yahoo Ticker | Asset / Benchmark Name | Units | Economic Role in GoldSniper |
| :--- | :--- | :--- | :--- | :--- |
| `dxy` | `DX-Y.NYB` | ICE US Dollar Index | Index | Primary currency denominator; inverse contemporaneous correlation (`r = -0.388`). |
| `us10y` | `^TNX` | 10-Year US Treasury Yield | Percent (`%`) | Benchmark long-term sovereign bond yield; measures opportunity cost of holding Gold. |
| `us5y` | `^FVX` | 5-Year US Treasury Yield | Percent (`%`) | Medium-term Treasury yield; paired with `us10y` to measure belly curve slope. |
| `us30y` | `^TYX` | 30-Year US Treasury Yield | Percent (`%`) | Long-bond yield; reflects long-horizon inflation and fiscal term premium. |
| `silver` | `SI=F` | COMEX Silver Futures | USD / troy oz | Monetary and industrial precious metal; strongest co-mover (`r = +0.769`) and ratio anchor. |
| `oil` | `CL=F` | NYMEX WTI Crude Oil Futures | USD / barrel | Global energy benchmark; strongest medium-term forward regime predictor (`r = -0.539` at 63d). |
| `copper` | `HG=F` | COMEX Copper Futures | USD / pound | Industrial growth bellwether; denominator for the `Gold/Copper` macro risk ratio. |
| `vix` | `^VIX` | CBOE Volatility Index | Index | Implied S&P 500 option volatility; captures equity tail-risk hedging demand. |
| `sp500` | `^GSPC` | S&P 500 Equity Index | Index | Broad US equity benchmark; trades until 4:00 PM ET (`+2.5h` after COMEX Gold settlement). |
| `gdx` | `GDX` | VanEck Gold Miners ETF | USD / share | NYSE-traded gold mining equities; closes at 4:00 PM ET, providing the strongest 1-day lead signal. |
| `eurusd` | `EURUSD=X` | EUR/USD Spot Rate | USD per EUR | Largest constituent weight (`57.6%`) of the US Dollar Index. |
| `usdjpy` | `JPY=X` | USD/JPY Spot Rate | JPY per USD | Key sovereign yield-differential and Yen carry-trade barometer. |

---

## 3. Clean Gold Table Schema (`data/gold_daily_5y_clean.csv`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `date` | `YYYY-MM-DD` | Trading session date (settled daily closes only). |
| `open` | `float` | Daily opening price (USD/oz, rounded to 2 decimal places). |
| `high` | `float` | Intraday high price (USD/oz). |
| `low` | `float` | Intraday low price (USD/oz). |
| `close` | `float` | Settled COMEX daily closing price (USD/oz). |
| `volume` | `int` | Contract volume (8 zero-volume anomalies imputed via 21-day centered rolling median). |
| `volume_imputed` | `bool` | `True` on the 8 dates where zero-volume glitches were imputed. |
| `log_close` | `float` | Natural logarithm of closing price $\ln(\text{close}_t)$. |
| `daily_return` | `float` | Simple percentage return $(\text{close}_t / \text{close}_{t-1}) - 1$. |
| `log_return` | `float` | Continuously compounded return $\ln(\text{close}_t / \text{close}_{t-1})$. |

*(Note: `data/gold_daily_5y_bday.csv` adds `is_holiday_filled: bool` to identify the 47 forward-filled weekday market holidays.)*

---

## 4. Engineered Daily Feature Matrix (`data/gold_daily_features.csv`, 70 Columns)

> **Causal Invariant:** Every predictor column below is evaluated at period $t-1$ (shifted by 1 trading day) so that row $t$ contains only information settled prior to trading day $t$.

### 4.1 Target Columns (2)
- `close`: Settled Gold closing price at day $t$ (`USD/oz`).
- `daily_return`: Simple daily return at day $t$.

### 4.2 Gold Lagged Levels & Moving Averages (5)
- `lag_1d`: Settled Gold close at $t-1$.
- `ema_3d`, `ema_5d`, `ema_10d`, `ema_20d`: Exponential moving averages (spans 3, 5, 10, 20 days) evaluated at $t-1$.

### 4.3 Gold Multi-Horizon Technical & Regime Spreads (6)
- `sma_3d_spread`: $(\text{close}_{t-1} / \text{SMA}_{3,t-1}) - 1$ (short-term overextension mean-reversion signal).
- `ema_5d_vs_10d`: $(\text{EMA}_{5,t-1} / \text{EMA}_{10,t-1}) - 1$ (short-term crossover momentum).
- `sma_20d_vs_100d`: $(\text{SMA}_{20,t-1} / \text{SMA}_{100,t-1}) - 1$ (medium-term trend regime).
- `ema_200d_vs_500d`: $(\text{EMA}_{200,t-1} / \text{EMA}_{500,t-1}) - 1$ (long-cycle secular extension indicator).
- `close_vs_ema20d`, `close_vs_ema200d`: Relative distance of $\text{close}_{t-1}$ from 20-day and 200-day EMAs.

### 4.4 Gold Lagged Returns, Volatility & Oscillators (11)
- `lag_1d_return`, `lag_2d_return`, `lag_3d_return`, `lag_7d_return`: Single-day returns lagged by 1, 2, 3, and 7 trading days.
- `lag_5d_return`, `lag_21d_return`: Cumulative 5-day (1-week) and 21-day (1-month) trailing returns at $t-1$.
- `vol_5d`, `vol_20d`: 5-day and 20-day rolling standard deviation of daily returns at $t-1$.
- `hl_range_lag1`: Intraday high-low range $(\text{high}_{t-1} - \text{low}_{t-1}) / \text{close}_{t-1}$.
- `rsi_14d_lag1`: 14-day Wilder/EMA Relative Strength Index at $t-1$, scaled to $[0, 1]$.
- `bb_pct_b_lag1`: 20-day Bollinger Band $\%B$ position at $t-1$, clipped to $[-1, 2]$.

### 4.5 12-Series Macro Levels, Returns/Changes & 20-Day EMA Spreads (38)
For each of the 12 macro series $m \in \{\text{dxy}, \text{us10y}, \text{us5y}, \text{us30y}, \text{silver}, \text{oil}, \text{copper}, \text{vix}, \text{sp500}, \text{gdx}, \text{eurusd}, \text{usdjpy}\}$:
- `{m}_lag1` (12 columns): Settled level at $t-1$.
- `{m}_ret_lag1` or `{m}_chg_lag1` (12 columns): 1-day percentage return at $t-1$ (or first difference in percentage points for yield series `us10y`, `us5y`, `us30y`).
- `{m}_vs_ema20d` (12 columns): Relative spread $(m_{t-1} / \text{EMA}_{20}(m)_{t-1}) - 1$.
- `oil_mom_20d`, `dxy_mom_20d` (2 columns): 20-day trailing percentage momentum for WTI Oil and DXY at $t-1$.

### 4.6 Cross-Asset Ratios, COMEX-NYSE Lead Signals & Yield Curve Slopes (8)
- `gold_silver_ratio_lag1`: $\text{Gold}_{t-1} / \text{Silver}_{t-1}$ (primary long-horizon value ratio).
- `gold_gdx_ratio_lag1`: $\text{Gold}_{t-1} / \text{GDX}_{t-1}$ (bullion-to-miners valuation ratio).
- `gold_gdx_ret_lag1`: 1-day percentage change in the `Gold/GDX` ratio at $t-1$ (**strongest 1-day return predictor**, $r = -0.1641, p < 10^{-8}$, capturing the 1:30 PM ET COMEX vs. 4:00 PM ET NYSE settlement gap).
- `gold_gdx_z5_lag1`: 5-day relative deviation of `Gold/GDX` from its 5-day rolling mean at $t-1$.
- `gold_copper_ratio_lag1`: $\text{Gold}_{t-1} / \text{Copper}_{t-1}$ (monetary vs. industrial metal ratio).
- `yc_10y_5y_lag1`: $\text{us10y}_{t-1} - \text{us5y}_{t-1}$ (10Y minus 5Y Treasury yield curve slope).
- `yc_30y_10y_lag1`: $\text{us30y}_{t-1} - \text{us10y}_{t-1}$ (30Y minus 10Y Treasury yield curve slope).

---

## 5. Engineered Weekly Feature Matrix (`data/gold_weekly_features.csv`, 67 Columns)

Resampled to Friday weekly closes (`W-FRI`) with every predictor strictly shifted by 1 week (`t-1w`):
- **Targets (2):** `close`, `weekly_return`
- **Weekly Gold Lags & Moving Averages (4):** `lag_1w`, `ema_2w`, `ema_4w`, `ema_8w`
- **Weekly Trend & Regime Spreads (5):** `ema_2w_vs_4w`, `ema_4w_vs_8w`, `close_vs_ema4w`, `close_vs_sma20w`, `ema_26w_vs_104w`
- **Weekly Lagged Returns, Volatility & RSI (9):** `lag_1w_return`, `lag_2w_return`, `lag_4w_return`, `lag_6w_return`, `lag_12w_return`, `mom_4w`, `vol_4w`, `hl_range_lag1w`, `rsi_14w_lag1w`
- **Weekly 12-Macro Levels, Changes & 8-Week EMA Spreads (40):** `{m}_lag1w` (12), `{m}_ret_lag1w` / `{m}_chg_lag1w` (12), `{m}_vs_ema8w` (12), `oil_mom_8w`, `dxy_mom_8w`, `silver_chg_lag6w`, `dxy_chg_lag12w`
- **Weekly Cross-Asset Ratios & Yield Curve Slopes (7):** `gold_silver_ratio_lag1w`, `gold_gdx_ratio_lag1w`, `gold_gdx_ret_lag1w`, `gold_copper_ratio_lag1w`, `yc_10y_5y_lag1w`, `yc_30y_10y_lag1w`
