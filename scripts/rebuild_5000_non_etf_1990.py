"""
Rebuild data for a large non-ETF universe with benchmark ETF comparators.

Default behavior:
- Universe: top 5000 non-ETF equities by liquidity
- Forced benchmark tickers: SPY, QQQ, IWM
- Date range: 1990-01-01 to user-provided end date
- Steps: universe,ohlcv,fundamentals
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild large non-ETF universe and raw data")
    parser.add_argument("--data_tag", type=str, default=None, help="Output run id (default: auto timestamp)")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument(
        "--steps",
        type=str,
        default="universe,ohlcv,fundamentals",
        help="Comma-separated pipeline steps",
    )
    parser.add_argument("--n_equities", type=int, default=5000, help="Number of non-ETF equities")
    parser.add_argument(
        "--benchmark_tickers",
        type=str,
        default="SPY,QQQ,IWM",
        help="Comma-separated benchmark tickers to force-include",
    )
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    benchmark_tickers = [
        t.strip().upper().replace(".", "-")
        for t in args.benchmark_tickers.split(",")
        if t.strip()
    ]

    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--n_equities",
        str(args.n_equities),
        "--start_date",
        "1990-01-01",
        "--end_date",
        args.end_date,
        "--steps",
        ",".join(steps),
        "--exclude_etfs",
        "--benchmark_tickers",
        ",".join(benchmark_tickers),
    ]
    if args.data_tag:
        cmd += ["--data_tag", args.data_tag]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
