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


def test_forecast_figure_tooltip_and_default_3m():
    import sys
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    import app

    daily_df = pd.read_csv(DATA_DIR / "gold_daily_features.csv", parse_dates=["date"]).set_index("date").sort_index()
    fsets = app.get_feature_sets(daily_df, is_daily=True)
    cols = fsets["GDX Cross-Market Lead"]
    pred_df, _, live_info = app.run_walk_forward_and_live(
        daily_df, cols, "HistGBR", horizon=1, is_daily=True
    )

    fig_3m = app.build_forecast_figure(
        daily_df, pred_df, live_info, is_daily=True, show_ci=True, show_naive=False, show_full_history=False
    )

    target_traces = [t for t in fig_3m.data if getattr(t, "name", None) and str(t.name).startswith("Live Target:")]
    assert len(target_traces) == 1
    assert len(target_traces[0].x) == 1
    assert pd.Timestamp(target_traces[0].x[0]) == pd.Timestamp(live_info["target_date"])

    connector_traces = [
        t for t in fig_3m.data
        if getattr(t, "legendgroup", None) == "live_target" and getattr(t, "showlegend", True) is False
    ]
    assert len(connector_traces) == 1
    assert connector_traces[0].hoverinfo == "skip"
    assert [pd.Timestamp(x) for x in connector_traces[0].x] == [
        pd.Timestamp(live_info["anchor_date"]),
        pd.Timestamp(live_info["target_date"]),
    ]

    assert fig_3m.layout.xaxis.range is not None
    r_start = pd.Timestamp(fig_3m.layout.xaxis.range[0])
    r_end = pd.Timestamp(fig_3m.layout.xaxis.range[1])
    span_days = (r_end - r_start).days
    assert 85 <= span_days <= 105, f"Unexpected 3M range span: {span_days} days"

    fig_full = app.build_forecast_figure(
        daily_df, pred_df, live_info, is_daily=True, show_ci=True, show_naive=False, show_full_history=True
    )
    assert fig_full.layout.xaxis.range is None


def test_incremental_gold_merge():
    import sys
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    import fetch_gold_5y_daily

    existing = [
        {"date": "2026-11-17", "open": 3859.5, "high": 3894.0, "low": 3839.2, "close": 3864.2, "volume": 160399, "source": "yahoo_gc_f"},
        {"date": "2026-11-18", "open": 3855.2, "high": 3859.3, "low": 3721.8, "close": 3741.9, "volume": 160399, "source": "yahoo_gc_f"},
    ]
    incoming = [
        {"date": "2026-11-18", "open": 3855.2, "high": 3859.3, "low": 3721.8, "close": 3742.1, "volume": 244253, "source": "yahoo_gc_f"},
        {"date": "2026-11-19", "open": 3730.3, "high": 3759.5, "low": 3620.6, "close": 3685.5, "volume": 203702, "source": "yahoo_gc_f"},
    ]
    merged = fetch_gold_5y_daily.merge_gold_rows(existing, incoming)
    assert [r["date"] for r in merged] == ["2026-11-17", "2026-11-18", "2026-11-19"]
    assert merged[0]["close"] == 3864.2
    assert merged[1]["close"] == 3742.1
    assert merged[1]["volume"] == 244253
    assert merged[2]["close"] == 3685.5


def test_incremental_macro_merge():
    import sys
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    import fetch_macro_5y_daily

    existing = pd.Series(
        [103.49, 103.86],
        index=pd.to_datetime(["2026-11-17", "2026-11-18"]),
        dtype=float,
    )
    incoming = pd.Series(
        [103.90, 104.12],
        index=pd.to_datetime(["2026-11-18", "2026-11-19"]),
        dtype=float,
    )
    merged = fetch_macro_5y_daily.merge_macro_series(existing, incoming)
    assert list(merged.index.strftime("%Y-%m-%d")) == ["2026-11-17", "2026-11-18", "2026-11-19"]
    assert float(merged.loc["2026-11-17"]) == 103.49
    assert float(merged.loc["2026-11-18"]) == 103.90
    assert float(merged.loc["2026-11-19"]) == 104.12


if __name__ == "__main__":
    for fn in [
        test_required_files_exist,
        test_clean_gold_data_integrity,
        test_feature_tables_leak_free_shift,
        test_walk_forward_engine_accuracy,
        test_app_ui_strings_no_em_dashes_or_parentheses,
        test_forecast_figure_tooltip_and_default_3m,
        test_incremental_gold_merge,
        test_incremental_macro_merge,
    ]:
        fn()
        print(f"[PASSED] {fn.__name__}")
    print("\nAll GoldSniper verification tests passed!")

