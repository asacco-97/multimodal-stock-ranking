import pandas as pd

from src.data_fetch.fetch_fundamentals_quarterly import merge_fundamentals_point_in_time


def test_report_date_not_after_price_date_after_merge():
    price = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15"]),
            "close": [10.0, 11.0, 12.0],
        }
    )
    f = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "quarter_end_date": ["2023-12-31", "2024-03-31"],
            "report_date": ["2024-02-01", "2024-04-30"],
            "revenue": [100.0, 120.0],
        }
    )
    merged = merge_fundamentals_point_in_time(price, f, valuation_df=None)
    valid = merged["report_date"].isna() | (merged["report_date"] <= merged["date"])
    assert bool(valid.all())
