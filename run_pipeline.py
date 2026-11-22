"""
GoldSniper Master Pipeline Orchestrator.

Executes the end-to-end GoldSniper data acquisition, cleaning, feature engineering,
exploratory correlation studies, multi-model walk-forward benchmarks, and >90% accuracy
optimization in a single reproducible command.

Usage:
    python run_pipeline.py              # Run full end-to-end pipeline
    python run_pipeline.py --data-only  # Only refresh data, clean, and rebuild features
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_STEPS = [
    ("1. Fetch 5-Year Daily Gold Futures (GC=F)", "fetch_gold_5y_daily.py"),
    ("2. Fetch 5-Year 12-Series Macro & Cross-Asset Universe", "fetch_macro_5y_daily.py"),
    ("3. Clean Gold OHLCV & Export Trading/Business-Day CSVs", "clean_gold_data.py"),
    ("4. Engineer Leak-Free Daily (70) & Weekly (67) Features", "build_gold_features.py"),
]

ANALYTICS_STEPS = [
    ("5. Short-Window Lag, SMA & EMA Correlation Study", "explore_predictors.py"),
    ("6. Medium & Long-Window MA/EMA Regime Study", "explore_bigger_ma_ema.py"),
    ("7. Macro Correlation & Granger Causality Analysis", "explore_macro_correlations.py"),
    ("8. 6-Tier Algorithm Walk-Forward Benchmark", "forecast_benchmark.py"),
    ("9. >90% Accuracy Walk-Forward Optimization Suite", "optimize_90_accuracy.py"),
]


def run_step(title: str, script_name: str, extra_args: list[str] | None = None) -> bool:
    script_path = BASE_DIR / script_name
    cmd = [sys.executable, str(script_path), *(extra_args or [])]
    print(f"\n{'=' * 78}")
    print(f"[GoldSniper] Running {title}")
    print(f"Script: {script_path.name}" + (f" {' '.join(extra_args)}" if extra_args else ""))
    print(f"{'=' * 78}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(BASE_DIR), check=False)
    elapsed = time.perf_counter() - t0
    status = "OK" if res.returncode == 0 else f"FAILED (exit code {res.returncode})"
    print(f"[GoldSniper] Completed {script_name} -> {status} in {elapsed:.2f}s")
    return res.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="GoldSniper End-to-End Pipeline Runner")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Run only data ingestion, cleaning, and feature engineering steps.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force a full 5-year refetch instead of incremental delta updates.",
    )
    args = parser.parse_args()

    steps = DATA_STEPS if args.data_only else (DATA_STEPS + ANALYTICS_STEPS)
    t_start = time.perf_counter()

    print("==============================================================================")
    print("  GoldSniper: Reproducible Gold Futures Data & Walk-Forward Forecasting Suite ")
    print("==============================================================================")

    fetch_scripts = {"fetch_gold_5y_daily.py", "fetch_macro_5y_daily.py"}
    for title, script in steps:
        extra = ["--full"] if (args.full and script in fetch_scripts) else None
        if not run_step(title, script, extra_args=extra):
            print(f"\n[GoldSniper] Pipeline halted due to failure in {script}.")
            return 1

    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 78}")
    print(f"[GoldSniper] ALL {len(steps)} PIPELINE STAGES SUCCEEDED in {total_time:.2f}s")
    print("Launch the interactive dashboard anytime with: streamlit run app.py")
    print(f"{'=' * 78}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
