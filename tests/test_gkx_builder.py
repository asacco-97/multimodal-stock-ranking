import numpy as np
import pandas as pd

from src.features.build_gkx_characteristics import (
    attach_gkx_to_daily,
    build_gkx_characteristics,
    generate_gkx_coverage_report,
)
from src.features.gkx_registry import GKX_94


def _sample_daily_df(n_days: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for ticker, seed in [("AAA", 1.0), ("BBB", 2.0)]:
        price = seed * 100.0
        shares = 1_000_000 if ticker == "AAA" else 2_000_000
        for i, d in enumerate(dates):
            price = price * (1 + (0.0005 if ticker == "AAA" else 0.0003))
            rows.append({
                "ticker": ticker,
                "date": d,
                "close": price,
                "open": price * 0.995,
                "high": price * 1.01,
                "low": price * 0.99,
                "volume": 100_000 + (i % 20) * 1000,
                "shares_outstanding": shares,
                "market_cap": price * shares,
                "total_assets": 10_000_000 + i * 5000,
                "total_equity": 6_000_000 + i * 3000,
                "total_debt": 2_000_000 + i * 1500,
                "total_cash": 500_000 + i * 300,
                "current_assets": 2_500_000 + i * 800,
                "current_liabilities": 1_200_000 + i * 600,
                "current_ratio": 2.0,
                "quick_ratio": 1.5,
                "revenue": 1_200_000 + i * 500,
                "net_income": 150_000 + i * 50,
                "operating_income": 180_000 + i * 45,
                "gross_profit": 420_000 + i * 100,
                "ebitda": 220_000 + i * 60,
                "inventory": 90_000 + i * 30,
                "receivables": 120_000 + i * 35,
                "depreciation": 30_000,
                "capex": 45_000,
                "sga_expense": 80_000,
                "research_and_development": 60_000,
                "operating_cash_flow": 170_000,
                "income_tax": 20_000,
                "industry": "Software" if ticker == "AAA" else "Semiconductors",
                "dividend_yield": 0.01,
            })
    return pd.DataFrame(rows)


def test_build_gkx_characteristics_has_all_columns():
    df = _sample_daily_df()
    gkx = build_gkx_characteristics(df)

    assert "ticker" in gkx.columns
    assert "month_end" in gkx.columns
    for c in GKX_94:
        assert c in gkx.columns
    assert len(gkx) > 0


def test_coverage_report_shape():
    df = _sample_daily_df()
    gkx = build_gkx_characteristics(df)
    cov = generate_gkx_coverage_report(gkx)
    assert len(cov) == 94
    assert {"gkx_name", "coverage_pct", "status"}.issubset(set(cov.columns))


def test_attach_back_to_daily_preserves_rows():
    df = _sample_daily_df()
    gkx = build_gkx_characteristics(df)
    merged = attach_gkx_to_daily(df, gkx)
    assert len(merged) == len(df)
    # At least one canonical feature should appear attached.
    assert "mom12m" in merged.columns
