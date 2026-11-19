# Contributing to GoldSniper

Thank you for your interest in contributing to **GoldSniper**! To keep the forecasting pipeline strictly leak-free, reproducible, and ready for academic or production publishing, please follow the standards below.

---

## 1. Core Engineering Rules

### 1.1 Zero Look-Ahead Leakage (`t-1` Invariant)
Every predictor stored in `data/gold_daily_features.csv` and `data/gold_weekly_features.csv` must be strictly known at the close of period `t-1` (using `.shift(1)` after rolling or exponential smoothing calculations) when predicting period `t` or horizon `t+h-1`.
- **Never** compute a rolling window, z-score, or standard scaler across the full dataset before splitting train and test periods.
- **Always** fit scalers and models strictly inside the expanding walk-forward training window `[:i - horizon + 1]`.

### 1.2 Predict Stationary Log-Returns, Reconstruct Price
Tree-based and regularized models extrapolate poorly when predicting non-stationary price levels during all-time-high breakouts (`$2,000 -> $5,318`).
- Always predict the horizon log-return:
  $$\hat{r}_{t,h} = \ln\left(\frac{\hat{P}_{t+h-1}}{P_{t-1}}\right)$$
- Reconstruct the dollar price forecast from the last settled anchor close:
  $$\hat{P}_{t+h-1} = P_{t-1} \cdot \exp(\hat{r}_{t,h})$$

### 1.3 Keyless, Reproducible Data Sources
All external series in `fetch_gold_5y_daily.py` and `fetch_macro_5y_daily.py` must remain accessible via keyless public endpoints so any researcher or reviewer can refresh the entire dataset with `python run_pipeline.py`.

### 1.4 Streamlit Storytelling & UI Style Rules (`app.py`)
When modifying `app.py`:
- **No Em Dashes or Parentheses in Visible UI Strings**: Keep all titles, labels, captions, trace names, narrative banners, and tooltips free of em dashes (`—`, `–`) and parentheses (`(`, `)`).
- **Decluttered Canvas with Native Tooltips**: Show only high-signal metrics and visuals on the main canvas; place formulas, methodological notes, and column definitions inside `help="..."` tooltips.

---

## 2. Development & Verification Workflow

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the data and feature pipeline**:
   ```bash
   python run_pipeline.py --data-only
   ```
3. **Run the automated verification suite**:
   ```bash
   pytest
   ```
4. **Preview the Streamlit dashboard**:
   ```bash
   streamlit run app.py
   ```
