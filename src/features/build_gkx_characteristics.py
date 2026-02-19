"""
Monthly GKX characteristic builder (OHLCV + SEC point-in-time proxy version).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.features.gkx_registry import GKX_94, get_gkx_registry


def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    if isinstance(b, pd.Series):
        return a / b.replace(0, np.nan)
    if pd.isna(b) or b == 0:
        return a * np.nan
    return a / b


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce").dt.as_unit("ns")


def _compute_market_return(daily: pd.DataFrame) -> pd.Series:
    """Cross-sectional equal-weight market return proxy by date."""
    return daily.groupby("date")["ret_1d"].transform("mean")


def _rolling_beta(group: pd.DataFrame, window: int = 252) -> pd.Series:
    r = group["ret_1d"]
    m = group["mkt_ret_1d"]
    cov = r.rolling(window).cov(m)
    var = m.rolling(window).var()
    return _safe_div(cov, var)


def _calc_derived_daily_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", group_keys=False)

    df["ret_1d"] = g["close"].pct_change()
    df["log_ret_1d"] = np.log1p(df["ret_1d"])
    df["dollar_vol"] = df["close"] * df["volume"]

    shares = df.get("shares_outstanding", np.nan)
    df["turn_proxy"] = _safe_div(df["volume"], shares)
    df["baspread_proxy"] = _safe_div(df["high"] - df["low"], df["close"])
    df["ill_proxy_daily"] = _safe_div(df["ret_1d"].abs(), df["dollar_vol"])

    # Momentum windows (trading-day approximations)
    df["mom1m"] = g["close"].transform(lambda s: _safe_div(s, s.shift(21)) - 1)
    df["mom6m"] = g["close"].transform(lambda s: _safe_div(s, s.shift(126)) - 1)
    df["mom12m"] = g["close"].transform(lambda s: _safe_div(s, s.shift(252)) - 1)
    df["mom36m"] = g["close"].transform(lambda s: _safe_div(s, s.shift(756)) - 1)
    df["chmom"] = df["mom6m"] - df["mom1m"]

    # Rolling risk and trading frictions
    df["retvol"] = g["ret_1d"].transform(lambda s: s.rolling(21).std())
    df["idiovol"] = g["ret_1d"].transform(lambda s: s.rolling(63).std())
    df["maxret"] = g["ret_1d"].transform(lambda s: s.rolling(21).max())
    df["std_dolvol"] = g["dollar_vol"].transform(lambda s: np.log1p(s).rolling(21).std())
    df["turn"] = g["turn_proxy"].transform(lambda s: s.rolling(21).mean())
    df["std_turn"] = g["turn_proxy"].transform(lambda s: s.rolling(21).std())
    df["zerotrade"] = g["volume"].transform(lambda s: (s <= 0).rolling(21).mean())
    df["dolvol"] = g["dollar_vol"].transform(lambda s: np.log1p(s).rolling(21).mean())
    df["baspread"] = g["baspread_proxy"].transform(lambda s: s.rolling(21).mean())
    df["ill"] = g["ill_proxy_daily"].transform(lambda s: s.rolling(21).mean())
    df["aeavol"] = g["volume"].transform(lambda s: np.log1p(s).rolling(63).std())

    # Beta to equal-weight market
    df["mkt_ret_1d"] = _compute_market_return(df)
    df["beta"] = g.apply(_rolling_beta).reset_index(level=0, drop=True)
    df["betasq"] = df["beta"] ** 2

    return df


def _build_monthly_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Take end-of-month row per ticker after daily rolling features are computed."""
    out = df.copy()
    out["month_end"] = out["date"] + pd.offsets.MonthEnd(0)
    out = out.sort_values(["ticker", "date"])
    out = out.groupby(["ticker", "month_end"], as_index=False).tail(1).reset_index(drop=True)
    return out


def _industry_adjust(series: pd.Series, industry: pd.Series, month_end: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"x": series, "industry": industry, "month_end": month_end})
    means = tmp.groupby(["month_end", "industry"])["x"].transform("mean")
    return series - means


def _compute_accounting_proxies(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.sort_values(["ticker", "month_end"]).copy()
    # Ensure optional source columns exist so groupby operations do not fail.
    optional_cols = [
        "market_cap", "shares_outstanding", "total_equity", "net_income", "ebitda", "revenue",
        "dividend_yield", "total_cash", "inventory", "receivables", "current_ratio", "quick_ratio",
        "total_debt", "total_assets", "depreciation", "gross_profit", "operating_income", "capex",
        "research_and_development", "real_estate_assets", "income_tax", "secured_debt",
        "convertible_debt", "employees", "sga_expense", "asset_turnover", "profit_margin",
        "operating_cash_flow", "industry", "ppe",
    ]
    for c in optional_cols:
        if c not in monthly.columns:
            monthly[c] = np.nan
    if "gross_margin" not in monthly.columns:
        monthly["gross_margin"] = _safe_div(monthly["gross_profit"], monthly["revenue"])
    if "profit_margin" not in monthly.columns:
        monthly["profit_margin"] = _safe_div(monthly["net_income"], monthly["revenue"])
    if "asset_turnover" not in monthly.columns:
        monthly["asset_turnover"] = _safe_div(monthly["revenue"], monthly["total_assets"])

    g = monthly.groupby("ticker", group_keys=False)

    mkt_cap = monthly.get("market_cap")
    if mkt_cap is None:
        mkt_cap = monthly.get("close", np.nan) * monthly.get("shares_outstanding", np.nan)
    monthly["mkt_cap_proxy"] = mkt_cap

    monthly["mvel1"] = np.log(monthly["mkt_cap_proxy"].replace(0, np.nan))
    monthly["bm"] = _safe_div(monthly.get("total_equity"), monthly["mkt_cap_proxy"])
    monthly["ep"] = _safe_div(monthly.get("net_income") * 4, monthly["mkt_cap_proxy"])
    monthly["cfp"] = _safe_div(monthly.get("ebitda") * 4, monthly["mkt_cap_proxy"])
    monthly["ps"] = _safe_div(monthly["mkt_cap_proxy"], monthly.get("revenue") * 4)
    monthly["sp"] = _safe_div(monthly.get("revenue") * 4, monthly["mkt_cap_proxy"])
    monthly["dy"] = monthly.get("dividend_yield", np.nan)

    monthly["cash"] = _safe_div(monthly.get("total_cash"), monthly.get("total_assets"))
    monthly["cashdebt"] = _safe_div(monthly.get("total_cash"), monthly.get("total_debt"))
    monthly["salecash"] = _safe_div(monthly.get("revenue") * 4, monthly.get("total_cash"))
    monthly["saleinv"] = _safe_div(monthly.get("revenue") * 4, monthly.get("inventory"))
    monthly["salerec"] = _safe_div(monthly.get("revenue") * 4, monthly.get("receivables"))
    monthly["currat"] = monthly.get("current_ratio", np.nan)
    monthly["quick"] = monthly.get("quick_ratio", np.nan)
    monthly["lev"] = _safe_div(monthly.get("total_debt"), monthly.get("total_assets"))
    monthly["depr"] = _safe_div(monthly.get("depreciation"), monthly.get("total_assets"))
    monthly["gma"] = _safe_div(monthly.get("gross_profit"), monthly.get("revenue"))
    monthly["operprof"] = _safe_div(monthly.get("operating_income"), monthly.get("total_equity"))
    monthly["roaq"] = _safe_div(monthly.get("net_income"), monthly.get("total_assets"))
    monthly["roeq"] = _safe_div(monthly.get("net_income"), monthly.get("total_equity"))
    monthly["roic"] = _safe_div(
        monthly.get("operating_income"),
        monthly.get("total_equity") + monthly.get("total_debt") - monthly.get("total_cash"),
    )

    monthly["agr"] = g["total_assets"].pct_change(12)
    monthly["invest"] = monthly["agr"]
    monthly["lgr"] = g["total_debt"].pct_change(12)
    monthly["grCAPX"] = g["capex"].pct_change(4)
    monthly["chinv"] = g["inventory"].pct_change(4)
    monthly["chcsho"] = g["shares_outstanding"].pct_change(12)
    monthly["pchcurrat"] = g["current_ratio"].pct_change(4)
    monthly["pchquick"] = g["quick_ratio"].pct_change(4)
    monthly["pchdepr"] = g["depreciation"].pct_change(4)
    monthly["sgr"] = g["revenue"].pct_change(4)
    monthly["egr"] = g["net_income"].pct_change(4)

    monthly["pchsale_pchinvt"] = g["revenue"].pct_change(4) - g["inventory"].pct_change(4)
    monthly["pchsale_pchrect"] = g["revenue"].pct_change(4) - g["receivables"].pct_change(4)
    monthly["pchsale_pchxsga"] = g["revenue"].pct_change(4) - g["sga_expense"].pct_change(4)
    monthly["pchsaleinv"] = _safe_div(monthly.get("revenue"), monthly.get("inventory"))
    monthly["pchsaleinv"] = monthly.groupby("ticker")["pchsaleinv"].pct_change(4)
    monthly["pchgm_pchsale"] = g["gross_margin"].pct_change(4) - g["revenue"].pct_change(4)

    monthly["acc"] = _safe_div(
        monthly.get("net_income") - monthly.get("operating_cash_flow"),
        monthly.get("total_assets"),
    )
    monthly["absacc"] = monthly["acc"].abs()
    monthly["pctacc"] = _safe_div(
        monthly.get("net_income") - monthly.get("operating_cash_flow"),
        monthly.get("net_income").abs(),
    )
    monthly["stdacc"] = g["acc"].transform(lambda s: s.rolling(12).std())
    monthly["stdcf"] = g["operating_cash_flow"].transform(lambda s: s.rolling(12).std())

    monthly["rd"] = _safe_div(monthly.get("research_and_development"), monthly.get("total_assets"))
    monthly["rd_sale"] = _safe_div(monthly.get("research_and_development"), monthly.get("revenue"))
    monthly["rd_mve"] = _safe_div(monthly.get("research_and_development"), monthly["mkt_cap_proxy"])
    monthly["realestate"] = _safe_div(monthly.get("real_estate_assets"), monthly.get("total_assets"))
    monthly["tang"] = _safe_div(
        monthly.get("total_cash")
        + 0.715 * monthly.get("receivables")
        + 0.547 * monthly.get("inventory")
        + 0.535 * monthly.get("ppe"),
        monthly.get("total_assets"),
    )
    monthly["tb"] = _safe_div(monthly.get("income_tax"), monthly.get("net_income").abs())
    monthly["secured"] = _safe_div(monthly.get("secured_debt"), monthly.get("total_debt"))
    monthly["securedind"] = (monthly["secured"] > 0).astype(float)

    # Weak proxies
    monthly["age"] = g.cumcount() / 12.0
    monthly["cashpr"] = _safe_div(monthly.get("total_cash"), monthly.get("close"))
    monthly["cinvest"] = _safe_div(monthly.get("capex"), monthly.get("total_assets"))
    monthly["convind"] = (monthly.get("convertible_debt").fillna(0) > 0).astype(float)
    monthly["divi"] = (g["dividend_yield"].diff(1) > 0).astype(float)
    monthly["divo"] = (g["dividend_yield"].diff(1) < 0).astype(float)
    monthly["ear"] = monthly.get("ret_1d")
    monthly["grltnoa"] = g["total_assets"].pct_change(12)
    monthly["hire"] = g["employees"].pct_change(4)
    monthly["ms"] = g["shares_outstanding"].pct_change(12)
    monthly["nincr"] = (g["net_income"].diff(4) > 0).astype(float)
    monthly["orgcap"] = g["sga_expense"].transform(lambda s: s.ewm(alpha=0.2, adjust=False).mean())
    monthly["pchcapx_ia"] = g["capex"].pct_change(4)
    monthly["pricedelay"] = 1 - monthly["beta"]
    monthly["rsup"] = g["revenue"].pct_change(4) - g["revenue"].pct_change(8)
    monthly["chatoia"] = g["asset_turnover"].diff(4)
    monthly["chempia"] = g["employees"].pct_change(4) - g["revenue"].pct_change(4)
    monthly["chpmia"] = g["profit_margin"].diff(4)
    monthly["chtx"] = g["income_tax"].pct_change(4)
    monthly["cfp_ia"] = monthly["cfp"]
    monthly["bm_ia"] = _industry_adjust(monthly["bm"], monthly["industry"], monthly["month_end"])
    monthly["mve_ia"] = _industry_adjust(monthly["mvel1"], monthly["industry"], monthly["month_end"])

    # Industry concentration and momentum proxies
    ind_total = monthly.groupby(["month_end", "industry"])["mkt_cap_proxy"].transform("sum")
    ind_share = _safe_div(monthly["mkt_cap_proxy"], ind_total)
    monthly["herf"] = monthly.groupby(["month_end", "industry"])["mkt_cap_proxy"].transform(
        lambda s: ((_safe_div(s, s.sum())) ** 2).sum()
    )
    monthly["indmom"] = monthly.groupby(["month_end", "industry"])["mom12m"].transform("mean")

    sin_industries = {"tobacco", "alcohol", "gaming", "casino", "coal", "oil", "weapon", "defense"}
    ind = monthly.get("industry", pd.Series(index=monthly.index, dtype="object")).astype(str).str.lower()
    monthly["sin"] = ind.apply(lambda x: float(any(k in x for k in sin_industries)))

    # Ensure no accidental carry from helper columns for GKX names only later.
    return monthly


def _extract_gkx_columns(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly[["ticker", "month_end"]].copy()
    for name in GKX_94:
        if name in monthly.columns:
            out[name] = monthly[name]
        else:
            out[name] = np.nan
    return out


def generate_gkx_coverage_report(gkx_monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Coverage report for canonical 94 GKX columns.
    """
    registry = get_gkx_registry()
    rows: List[Dict] = []
    n = len(gkx_monthly_df)
    for _, r in registry.iterrows():
        name = r["gkx_name"]
        non_null = gkx_monthly_df[name].notna().sum() if name in gkx_monthly_df.columns else 0
        rows.append({
            "gkx_name": name,
            "alias": r["alias"],
            "frequency": r["frequency"],
            "status": r["status"],
            "rows_total": n,
            "rows_non_null": int(non_null),
            "coverage_pct": float(non_null / n * 100.0) if n > 0 else 0.0,
            "missing_reason": "" if non_null > 0 else "missing_inputs_or_not_available_in_source",
        })
    return pd.DataFrame(rows).sort_values(["coverage_pct", "gkx_name"], ascending=[False, True])


def build_gkx_characteristics(df: pd.DataFrame, asof: str = "month_end") -> pd.DataFrame:
    """
    Build monthly GKX characteristics from merged daily point-in-time data.

    Args:
        df: Daily merged dataset with PIT fundamentals.
        asof: Currently supports "month_end".
    """
    if asof != "month_end":
        raise ValueError("Only asof='month_end' is supported.")

    required = {"ticker", "date", "close", "high", "low", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for GKX build: {sorted(missing)}")

    work = df.copy()
    work["date"] = _ensure_datetime(work, "date")
    if "report_date" in work.columns:
        work["report_date"] = _ensure_datetime(work, "report_date")

    work = _calc_derived_daily_inputs(work)
    monthly = _build_monthly_snapshot(work)
    monthly = _compute_accounting_proxies(monthly)

    gkx = _extract_gkx_columns(monthly)
    return gkx


def attach_gkx_to_daily(df_daily: pd.DataFrame, gkx_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Join monthly GKX values to daily rows by ticker/month_end.
    """
    out = df_daily.copy()
    out["date"] = _ensure_datetime(out, "date")
    out["month_end"] = out["date"] + pd.offsets.MonthEnd(0)
    merged = out.merge(gkx_monthly, on=["ticker", "month_end"], how="left")
    return merged


def build_and_attach_gkx(df_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
    1) Build monthly GKX characteristics
    2) Generate coverage report
    3) Attach monthly GKX back to daily dataset
    """
    gkx_monthly = build_gkx_characteristics(df_daily, asof="month_end")
    coverage = generate_gkx_coverage_report(gkx_monthly)
    attached = attach_gkx_to_daily(df_daily, gkx_monthly)
    return attached, gkx_monthly, coverage
