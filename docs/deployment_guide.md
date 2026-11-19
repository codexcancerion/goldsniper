# GoldSniper Deployment & Publishing Guide

**GoldSniper** is packaged for one-command local execution, Docker containerization, or direct publishing to **Streamlit Community Cloud**.

---

## 1. Local Quickstart

### Prerequisites
- Python `3.10+`
- Internet connection (only needed when refreshing live Yahoo Finance data)

### Step-by-Step Setup
```bash
# 1. Navigate to the GoldSniper repository
cd "118 - DATA VISUALIZATION AND ANALYTICS/GOLD ANALYTICS"

# 2. Install required dependencies
pip install -r requirements.txt

# 3. (Optional) Refresh all 5-year market data, clean, rebuild features, and run benchmarks
python run_pipeline.py

# 4. Launch the GoldSniper Streamlit application
streamlit run app.py
```

The application will open automatically at `http://localhost:8501` with the pre-configured dark-gold institutional theme (`.streamlit/config.toml`).

---

## 2. Publishing to Streamlit Community Cloud

Because all cleaned datasets (`data/*.csv`) and benchmark summaries (`outputs/*.csv`) are self-contained and lightweight, **GoldSniper** deploys instantaneously on Streamlit Community Cloud without requiring any external database or paid API keys:

1. Push the **GoldSniper** repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) and click **Create app**.
3. Select your repository, branch, and set **Main file path** to `app.py`.
4. Click **Deploy**.
   - Streamlit Cloud automatically reads `requirements.txt` and `.streamlit/config.toml`.
   - Users viewing the published app can explore all 4 storytelling chapters, hover over interactive tooltips, export walk-forward forecast ledgers as CSV, and click **Refresh Live Market Data** in the sidebar to pull the latest Yahoo Finance closes on demand.

---

## 3. Running Automated Verification Tests

Before publishing a new release, run the automated pytest suite to verify data integrity, zero look-ahead leakage, walk-forward forecasting accuracy, and UI storytelling rules:

```bash
pytest -v
```
