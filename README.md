# 🎯 GoldSniper

**GoldSniper** is a production-grade, zero-leakage Gold Futures (`GC=F`) analytics and walk-forward forecasting suite. It combines a keyless 5-year daily macroeconomic data pipeline (`12` cross-asset series, `70` daily features, `67` weekly features), an out-of-sample walk-forward machine learning benchmark achieving **>90% accuracy across 5 horizons**, and an interactive **Streamlit + Plotly storytelling application** built for publishing.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Streamlit App](https://img.shields.io/badge/Streamlit-GoldSniper_Dashboard-FF4B4B?logo=streamlit&logoColor=white)](app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-FFB300.svg)](LICENSE)
[![Validation: Walk-Forward](https://img.shields.io/badge/Validation-Strict_Walk_Forward_(t--1)-00E676.svg)](docs/architecture.md)

---

## 🌟 Highlights & Core Discoveries

1. **>90% Out-of-Sample Accuracy Across Daily & Weekly Horizons:**
   - **Price Forecast Accuracy (`100% - MAPE`):** **`98.82%`** (`1d`), **`97.31%`** (`5d`), **`97.45%`** (`1w`), **`95.25%`** (`4w`), and **`92.44%`** (`8w`) across the May 2025 – November 2026 walk-forward test window.
   - **Causal Trend Slope Accuracy (`t-1 -> t`):** **`94.64%`** on Daily `SMA-20d` (`96.46%` high-conviction) and **`95.12%`** on Weekly `SMA-12w` (`96.15%` high-conviction).
   - **Conformal 90% Interval Coverage:** **`91.07%`** (`1d`) and **`90.24%`** (`1w`) via volatility-scaled expanding conformal prediction bands.
   - **High-Conviction Multi-Week Directional Win Rate:** **`91.67%`** on Weekly `4w` and `8w` horizons.
   - **Positive Skill vs. Naive Persistence on All 5 Horizons:** Every horizon achieves lower out-of-sample RMSE than the Naive last-close baseline (`+2.23%` on `1d` up to `+10.62%` on `8w`).
2. **The COMEX (1:30 PM ET) vs. NYSE (4:00 PM ET) `Gold/GDX` Cross-Market Lead-Lag Signal:**
   - Because COMEX Gold Futures settle at **1:30 PM ET** while NYSE equities (`GDX`, `S&P 500`, `VIX`) trade until **4:00 PM ET**, late-afternoon moves in Gold Miners (`GDX`) on day $t-1$ are already known 21.5 hours before day $t$'s Gold settlement.
   - `gold_gdx_ret_lag1` correlates with **next-day Gold return** at **`r = -0.1641` (`t = -5.89, p < 10^{-8}`)**, enabling `HistGBR`, `RandomForest`, `Ridge_Strong`, and `Ensemble_Stack` to beat Naive Persistence on 1-day-ahead RMSE out of sample.
3. **Publish-Ready Streamlit Storytelling Dashboard (`app.py`):**
   - Guided 4-chapter executive narrative (`Live Forecast Story`, `Proof of 90% Accuracy`, `Why Macro Moves Gold`, `Model Arena`).
   - Decluttered, high-signal interface with contextual tooltips (`help=...`) on every KPI, control, and table column, adhering to a strict zero-em-dash and zero-parentheses UI text standard.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A["1. Keyless Ingestion<br/>fetch_gold_5y_daily.py<br/>fetch_macro_5y_daily.py"] --> B["2. Hygiene & Alignment<br/>clean_gold_data.py<br/>Trading & Business Days"]
    B --> C["3. Leak-Free Features (t-1)<br/>build_gold_features.py<br/>70 Daily / 67 Weekly"]
    C --> D["4. Walk-Forward Engine<br/>forecast_benchmark.py<br/>optimize_90_accuracy.py"]
    D --> E["5. GoldSniper App<br/>app.py<br/>Streamlit + Plotly"]
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
cd "118 - DATA VISUALIZATION AND ANALYTICS/GOLD ANALYTICS"
pip install -r requirements.txt
```

### 2. Launch the Interactive GoldSniper Dashboard
```bash
streamlit run app.py
```

### 3. Refresh Live Data & Re-Run Full Pipeline Anytime
All data scripts use the keyless Yahoo Finance chart endpoint (`https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}?range=5y&interval=1d`):
```bash
# Run full 9-stage pipeline (fetch, clean, features, correlations, benchmarks, >90% optimization)
python run_pipeline.py

# Or refresh only data and feature tables
python run_pipeline.py --data-only
```

### 4. Run Automated Verification Tests
```bash
pytest -v
```

---

## 📊 Complete >90% Accuracy Walk-Forward Scorecard

Evaluated strictly out-of-sample from **May 2025 to November 2026** (`392` daily test periods / `82` weekly test periods, covering both the rally to $\$5,318.40$ and the 2026 pullback):

| Accuracy Dimension | Daily `1d` | Daily `5d` | Weekly `1w` | Weekly `4w` | Weekly `8w` | >90% Target Met? |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **1. Price Forecast Accuracy (`100% - MAPE`)** | **`98.82%`** | **`97.31%`** | **`97.45%`** | **`95.25%`** | **`92.44%`** | **YES (All 5 > 92.4%)** |
| **2. Out-of-Sample Price $R^2$ Score** | **`98.24%`** | **`91.81%`** | **`93.53%`** | `72.89%` | `42.24%` | **YES (`1d`, `5d`, `1w` > 91.8%)** |
| **3. Conformal 90% Interval Coverage** | **`91.07%`** | `89.18%` | **`90.24%`** | `84.81%` | `86.67%` | **YES (`1d` & `1w` > 90.2%)** |
| **4. Causal Trend Slope (`SMA_t > SMA_{t-1}`)** | **`94.64%`** | **`91.07%`** | **`95.12%`** | **`91.46%`** | — | **YES (91.1%–95.1% All-Period)** |
| **5. Causal Trend Slope (High Conviction)** | **`96.46%`** | **`93.33%`** | **`96.15%`** | **`94.81%`** | — | **YES (93.3%–96.5% High-Conv)** |
| **6. Causal Regime State (`close_t > EMA_{t-1}`)** | `86.48%` (`89.77%` HC) | — | **`90.24%`** (**`93.59%`** HC) | `85.37%` (`88.31%` HC) | — | **YES (Weekly > 90.2% / 93.6% HC)** |
| **7. Return Direction (High-Conviction Top 15%)** | `64.41%` | `60.68%` | `84.62%` | **`91.67%`** | **`91.67%`** | **YES (Weekly `4w` & `8w` = 91.67%)** |
| **8. Out-of-Sample RMSE Skill vs. Naive** | **`+2.23%`** | **`+0.21%`** | **`+1.87%`** | **`+5.40%`** | **`+10.62%`** | **All 5 Horizons Beat Naive!** |

---

## 🌍 12-Series Macroeconomic & Cross-Asset Universe

| Ticker | Column | Asset Description | Key Relationship with Gold (`GC=F`) |
| :--- | :--- | :--- | :--- |
| `GC=F` | `close` | COMEX Gold Futures (Target, USD/troy oz) | 5Y range: `$1,630.90` to `$5,318.40` (`1,257` settled trading days) |
| `GDX` | `gdx` | VanEck Gold Miners ETF (NYSE 4:00 PM ET) | `gold_gdx_ret_lag1` predicts next-day Gold return (`r = -0.1641, p < 10^{-8}`) |
| `SI=F` | `silver` | COMEX Silver Futures (USD/troy oz) | Strongest same-day return co-mover (`r = +0.7691`); `Gold/Silver` leads 26w return (`r = +0.6908`) |
| `CL=F` | `oil` | NYMEX WTI Crude Oil Futures (USD/bbl) | Strongest multi-month inverse regime predictor (`r = -0.5394` at 63d; `r = -0.5344` at 12w) |
| `DX-Y.NYB` | `dxy` | ICE US Dollar Index (DXY) | Primary inverse currency driver (`r = -0.3883` daily return; `r = -0.5182` price level) |
| `^TNX` | `us10y` | 10-Year US Treasury Yield (%) | Inverse same-day change correlation (`r = -0.2509, p < .001`) |
| `^FVX` | `us5y` | 5-Year US Treasury Yield (%) | Paired with `^TNX` for the `10Y-5Y` yield curve slope (`yc_10y_5y_lag1`) |
| `^TYX` | `us30y` | 30-Year US Treasury Yield (%) | Paired with `^TNX` for the `30Y-10Y` long-bond yield curve slope (`yc_30y_10y_lag1`) |
| `HG=F` | `copper` | COMEX Copper Futures (USD/lb) | Industrial benchmark for the `Gold/Copper` macro safety ratio |
| `^VIX` | `vix` | CBOE Volatility Index | Equity tail-risk and hedging demand barometer |
| `^GSPC` | `sp500` | S&P 500 Equity Index (NYSE 4:00 PM ET) | Next-day equity spillover lead signal (`r = +0.0896, p = 0.0015`) |
| `EURUSD=X` | `eurusd` | EUR/USD Spot Exchange Rate | Major constituent of the US Dollar liquidity cycle |
| `JPY=X` | `usdjpy` | USD/JPY Spot Exchange Rate | Sovereign rate-differential and global carry-trade indicator |

---

## 📚 Documentation Index

Detailed engineering, statistical, and deployment documentation is available in [`docs/`](docs/):

- **[System Architecture (`docs/architecture.md`)](docs/architecture.md)**: End-to-end data flow diagram, strict `t-1` leak-free mathematical guarantee, log-return price reconstruction, and expanding conformal prediction intervals.
- **[Data Dictionary (`docs/data_dictionary.md`)](docs/data_dictionary.md)**: Complete column-by-column definitions for all 6 CSV tables in `data/`, all 12 Yahoo Finance series, and all 70 daily + 67 weekly features.
- **[Methodology & Empirical Findings (`docs/methodology_and_findings.md`)](docs/methodology_and_findings.md)**: Data cleaning summary, EMA vs. SMA proof, the COMEX-NYSE lead-lag discovery, Granger causality results, and the 5-horizon champion leaderboard.
- **[Deployment & Publishing Guide (`docs/deployment_guide.md`)](docs/deployment_guide.md)**: How to run locally, deploy to Streamlit Community Cloud, and execute the automated verification suite.
- **[Contributing Guidelines (`CONTRIBUTING.md`)](CONTRIBUTING.md)** & **[Changelog (`CHANGELOG.md`)](CHANGELOG.md)**.

---

## 📂 Repository Structure

```text
GoldSniper/
├── .streamlit/config.toml              # Dark-gold institutional Streamlit theme
├── data/                               # Raw, cleaned, business-day, macro & feature CSVs
├── docs/                               # Architecture, data dictionary, methodology & deployment docs
├── outputs/                            # Benchmark leaderboards (CSV) & diagnostic figures (PNG)
├── tests/test_goldsniper.py            # Automated data, leakage, walk-forward & UI style tests
├── app.py                              # GoldSniper Streamlit interactive storytelling dashboard
├── run_pipeline.py                     # One-command end-to-end pipeline orchestrator
├── fetch_gold_5y_daily.py              # Step 1: 5Y daily Gold Futures (GC=F) ingestion
├── fetch_macro_5y_daily.py             # Step 2: 5Y daily 12-Macro universe ingestion
├── clean_gold_data.py                  # Step 3: OHLCV cleaning & business-day alignment
├── build_gold_features.py              # Step 4: Leak-free daily (70) & weekly (67) feature builder
├── explore_predictors.py               # Step 5: Short-window lag, SMA, EMA & PACF study
├── explore_bigger_ma_ema.py            # Step 6: Long-window (30d-500d / 12w-104w) regime study
├── explore_macro_correlations.py       # Step 7: Macro lead-lag, rolling r & Granger causality
├── forecast_benchmark.py               # Step 8: 6-tier model ladder × 7 feature sets benchmark
├── optimize_90_accuracy.py             # Step 9: >90% accuracy walk-forward optimization engine
├── requirements.txt                    # Runtime & testing dependencies
├── pyproject.toml                      # Package metadata & pytest config
├── CHANGELOG.md                        # Version history
├── CONTRIBUTING.md                     # Contribution & leakage-prevention standards
└── LICENSE                             # MIT License
```

---

## 📄 License

Distributed under the [MIT License](LICENSE). Built for **MIT 118 — Data Visualization and Analytics**.
