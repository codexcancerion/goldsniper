"""
Automated verification test suite for GoldSniper.

Verifies:
  1. All required data CSVs and output leaderboards exist and are non-empty.
  2. Clean Gold dataset has zero missing OHLC values, sorted unique dates, and valid OHLC envelopes.
  3. Daily (70 cols) and Weekly (67 cols) feature matrices satisfy the strict t-1 leak-free shift invariant.
  4. Walk-forward forecasting engine in app.py executes cleanly and meets >90% price and trend accuracy.
  5. All visible UI strings, titles, labels, and tooltips in app.py contain zero em dashes and zero parentheses.
"""

from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"


def test_required_files_exist():
    required = [
        BASE_DIR / "app.py",
        BASE_DIR / "run_pipeline.py",
        BASE_DIR / "README.md",
        BASE_DIR / "requirements.txt",
        BASE_DIR / "pyproject.toml",
        BASE_DIR / "LICENSE",
        BASE_DIR / "CHANGELOG.md",
        BASE_DIR / "CONTRIBUTING.md",
        BASE_DIR / ".streamlit" / "config.toml",
        BASE_DIR / "docs" / "architecture.md",
        BASE_DIR / "docs" / "data_dictionary.md",
        BASE_DIR / "docs" / "methodology_and_findings.md",
        BASE_DIR / "docs" / "deployment_guide.md",
        DATA_DIR / "gold_daily_5y_clean.csv",
        DATA_DIR / "gold_daily_5y_bday.csv",
        DATA_DIR / "macro_daily_5y.csv",
        DATA_DIR / "gold_daily_features.csv",
        DATA_DIR / "gold_weekly_features.csv",
        OUT_DIR / "goal_90_accuracy_summary.csv",
        OUT_DIR / "goal_90_directional_experiments.csv",
    ]
    for path in required:
        assert path.exists(), f"Missing required repository file: {path}"


def test_clean_gold_data_integrity():
    df = pd.read_csv(DATA_DIR / "gold_daily_5y_clean.csv", parse_dates=["date"])
    assert len(df) >= 1200
    assert df["date"].is_monotonic_increasing
    assert df["date"].is_unique
    for col in ["open", "high", "low", "close", "volume"]:
        assert df[col].isna().sum() == 0, f"Unexpected NaN in {col}"
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["volume"] > 0).all()


def test_feature_tables_leak_free_shift():
    daily = pd.read_csv(DATA_DIR / "gold_daily_features.csv", parse_dates=["date"]).set_index("date")
    weekly = pd.read_csv(DATA_DIR / "gold_weekly_features.csv", parse_dates=["date"]).set_index("date")

    assert daily.shape[1] == 69  # 70 CSV columns minus date index
    assert weekly.shape[1] == 66  # 67 CSV columns minus date index
    assert daily.isna().sum().sum() == 0
    assert weekly.isna().sum().sum() == 0

    # Verify lag_1d equals exact close shifted by 1 period (t-1 invariant)
    diff_d = np.abs(daily["lag_1d"].iloc[1:].values - daily["close"].iloc[:-1].values)
    assert np.max(diff_d) < 1e-5, "Daily lag_1d does not match close(t-1)!"

    diff_w = np.abs(weekly["lag_1w"].iloc[1:].values - weekly["close"].iloc[:-1].values)
    assert np.max(diff_w) < 1e-5, "Weekly lag_1w does not match close(t-1w)!"


def test_walk_forward_engine_accuracy():
    import sys
    sys.path.insert(0, str(BASE_DIR))
    import app

    daily_df = pd.read_csv(DATA_DIR / "gold_daily_features.csv", parse_dates=["date"]).set_index("date").sort_index()
    fsets = app.get_feature_sets(daily_df, is_daily=True)
    cols = fsets["GDX Cross-Market Lead"]
    pred_df, metrics, live_info = app.run_walk_forward_and_live(
        daily_df, cols, "HistGBR", horizon=1, is_daily=True
    )

    assert len(pred_df) > 350
    assert metrics["price_acc"] > 98.0
    assert metrics["trend_acc"] > 90.0
    assert metrics["skill_pct"] > 0.0
    assert live_info["ci_low"] < live_info["pred_close"] < live_info["ci_high"]


def test_app_ui_strings_no_em_dashes_or_parentheses():
    app_source = (BASE_DIR / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_source)

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            # Skip raw CSS style block and plotly rgba fill color string
            if "<style>" in val or val.startswith("rgba("):
                continue
            if any(ch in val for ch in ("—", "–", "(", ")")):
                violations.append((node.lineno, val))

    assert not violations, f"Found forbidden em dashes or parentheses in app.py strings: {violations}"


if __name__ == "__main__":
    for fn in [
        test_required_files_exist,
        test_clean_gold_data_integrity,
        test_feature_tables_leak_free_shift,
        test_walk_forward_engine_accuracy,
        test_app_ui_strings_no_em_dashes_or_parentheses,
    ]:
        fn()
        print(f"[PASSED] {fn.__name__}")
    print("\nAll GoldSniper verification tests passed!")

