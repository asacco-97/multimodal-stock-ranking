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


def _coalesce_from_candidates(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """
    Return the first non-null value across candidate columns, row-wise.
    """
    out = pd.Series(np.nan, index=df.index)
    for c in candidates:
        if c in df.columns:
            out = out.combine_first(df[c])
    return out


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

    shares = _coalesce_from_candidates(df, ["shares_outstanding", "shares_outstanding_x", "shares_outstanding_y"])
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
    # Normalize source column names that can vary by fundamentals provider/version.
    # This preserves any existing canonical columns and only fills gaps from aliases.
    source_aliases: Dict[str, List[str]] = {
        "market_cap": ["market_cap", "marketCap"],
        "shares_outstanding": ["shares_outstanding", "shares_outstanding_x", "shares_outstanding_y", "sharesOutstanding"],
        "total_equity": ["total_equity", "book_value", "stockholders_equity", "shareholders_equity"],
        "net_income": ["net_income", "netIncome"],
        "ebitda": ["ebitda"],
        "revenue": ["revenue", "total_revenue", "totalRevenue"],
        "dividend_yield": ["dividend_yield", "trailing_annual_dividend_yield", "trailingAnnualDividendYield"],
        "total_cash": ["total_cash", "cash_and_cash_equivalents", "cash"],
        "inventory": ["inventory", "inventories"],
        "receivables": ["receivables", "accounts_receivable", "accounts_receivable_net"],
        "current_assets": ["current_assets"],
        "current_liabilities": ["current_liabilities"],
        "current_ratio": ["current_ratio", "currentRatio"],
        "quick_ratio": ["quick_ratio", "quickRatio"],
        "total_debt": ["total_debt", "totalDebt"],
        "total_assets": ["total_assets", "totalAssets"],
        "depreciation": ["depreciation", "depreciation_and_amortization", "depreciationAndAmortization"],
        "gross_profit": ["gross_profit", "grossProfit"],
        "operating_income": ["operating_income", "operatingIncome"],
        "capex": ["capex", "capital_expenditures", "capitalExpenditures"],
        "research_and_development": ["research_and_development", "rnd", "r_and_d_expense", "researchDevelopment"],
        "real_estate_assets": ["real_estate_assets", "realEstateAssets"],
        "income_tax": ["income_tax", "income_tax_expense", "incomeTaxExpense"],
        "secured_debt": ["secured_debt"],
        "convertible_debt": ["convertible_debt"],
        "employees": ["employees", "employee_count"],
        "sga_expense": ["sga_expense", "selling_general_and_administrative", "sellingGeneralAdministrative"],
        "asset_turnover": ["asset_turnover"],
        "profit_margin": ["profit_margin"],
        "operating_cash_flow": ["operating_cash_flow", "cash_flow_from_operations", "operatingCashFlow"],
        "industry": ["industry"],
        "ppe": ["ppe", "property_plant_equipment", "property_plant_and_equipment"],
    }
    for canonical, candidates in source_aliases.items():
        monthly[canonical] = _coalesce_from_candidates(monthly, candidates)

    # Derive helper ratios if not supplied by source data.
    if "gross_margin" not in monthly.columns or monthly["gross_margin"].notna().sum() == 0:
        monthly["gross_margin"] = _safe_div(monthly["gross_profit"], monthly["revenue"])
    if monthly["profit_margin"].notna().sum() == 0:
        monthly["profit_margin"] = _safe_div(monthly["net_income"], monthly["revenue"])
    if monthly["asset_turnover"].notna().sum() == 0:
        monthly["asset_turnover"] = _safe_div(monthly["revenue"], monthly["total_assets"])
    if monthly["current_ratio"].notna().sum() == 0:
        monthly["current_ratio"] = _safe_div(monthly["current_assets"], monthly["current_liabilities"])
    if monthly["quick_ratio"].notna().sum() == 0:
        monthly["quick_ratio"] = _safe_div(
            monthly["current_assets"] - monthly["inventory"].fillna(0),
            monthly["current_liabilities"],
        )

    g = monthly.groupby("ticker", group_keys=False)

    market_cap_fallback = monthly.get("close", np.nan) * monthly["shares_outstanding"]
    monthly["mkt_cap_proxy"] = monthly["market_cap"].combine_first(market_cap_fallback)

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
    monthly["roavol"] = g["roaq"].transform(lambda s: s.rolling(12).std())
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
    ind = (
        monthly.get("industry", pd.Series(index=monthly.index, dtype="object"))
        .astype("string")
        .fillna("")
        .str.lower()
    )
    sin_pattern = "|".join(sorted(sin_industries))
    monthly["sin"] = ind.str.contains(sin_pattern, regex=True, na=False).astype(float)

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


def _add_next_eom_return_target(gkx_monthly: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Add next end-of-month return target at monthly frequency.
    return_t+1m = close_{t+1 month-end} / close_{t month-end} - 1
    """
    out = gkx_monthly.copy()
    px = monthly[["ticker", "month_end", "close"]].copy()
    px = px.sort_values(["ticker", "month_end"])
    px["return_eom_t+1"] = px.groupby("ticker")["close"].shift(-1) / px["close"] - 1
    out = out.merge(px[["ticker", "month_end", "return_eom_t+1"]], on=["ticker", "month_end"], how="left")
    return out


def _add_cross_sectional_rank_normalization(gkx_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    GKX-style preprocessing at each month:
    1) Impute missing characteristic values with cross-sectional median.
    2) Add cross-sectional percentile rank and [-1, 1] normalized versions.
    """
    out = gkx_monthly.copy()

    # Ensure canonical raw columns exist first.
    for name in GKX_94:
        if name not in out.columns:
            out[name] = np.nan

    # Build transformed columns in a side dict and concatenate once to avoid
    # DataFrame fragmentation from repeated column insertion.
    transformed = {}
    grouped = out.groupby("month_end", sort=False)

    for name in GKX_94:
        # Cross-sectional median imputation by month.
        cs_median = grouped[name].transform("median")
        out[name] = out[name].fillna(cs_median)

        # Cross-sectional rank in (0, 1], then map to [-1, 1].
        cs_rank = grouped[name].rank(method="average", pct=True)
        transformed[f"{name}_csrank"] = cs_rank.astype(np.float32)
        transformed[f"{name}_csnorm"] = (2.0 * cs_rank - 1.0).astype(np.float32)

    if transformed:
        out = pd.concat([out, pd.DataFrame(transformed, index=out.index)], axis=1)

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
    gkx = _add_next_eom_return_target(gkx, monthly)
    gkx = _add_cross_sectional_rank_normalization(gkx)
    return gkx


def attach_gkx_to_daily(
    df_daily: pd.DataFrame,
    gkx_monthly: pd.DataFrame,
    lag_months: int = 1,
    include_transformed: bool = False,
) -> pd.DataFrame:
    """
    Join monthly GKX values to daily rows by ticker with a lagged month-end key.

    `lag_months=1` is the safe default for no-lookahead:
    each daily row at date t receives characteristics from the prior month-end.
    """
    if lag_months < 1:
        raise ValueError("lag_months must be >= 1 to avoid look-ahead leakage.")

    out = df_daily.copy()
    out["date"] = _ensure_datetime(out, "date")
    out["gkx_month_end"] = out["date"] + pd.offsets.MonthEnd(-lag_months)

    # Attach canonical raw GKX columns by default.
    # Optional transformed columns are available but omitted by default for memory efficiency.
    attach_cols = list(GKX_94)
    if include_transformed:
        attach_cols += [c for c in gkx_monthly.columns if c.endswith("_csrank") or c.endswith("_csnorm")]
    attach_cols = [c for c in attach_cols if c in gkx_monthly.columns]

    gkx_attach = gkx_monthly[["ticker", "month_end"] + attach_cols].copy()

    # Avoid _x/_y suffixes by preferring attached GKX values on overlapping column names.
    overlap = [c for c in attach_cols if c in out.columns]
    if overlap:
        out = out.drop(columns=overlap)
    merged = out.merge(
        gkx_attach,
        left_on=["ticker", "gkx_month_end"],
        right_on=["ticker", "month_end"],
        how="left",
    )
    # Keep a single canonical key name in the output.
    merged = merged.drop(columns=["month_end"]).rename(columns={"gkx_month_end": "month_end"})
    return merged


def build_and_attach_gkx(
    df_daily: pd.DataFrame,
    lag_months: int = 1,
    include_transformed: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
    1) Build monthly GKX characteristics
    2) Generate coverage report
    3) Attach monthly GKX back to daily dataset
    """
    gkx_monthly = build_gkx_characteristics(df_daily, asof="month_end")
    coverage = generate_gkx_coverage_report(gkx_monthly)
    attached = attach_gkx_to_daily(
        df_daily,
        gkx_monthly,
        lag_months=lag_months,
        include_transformed=include_transformed,
    )
    return attached, gkx_monthly, coverage
