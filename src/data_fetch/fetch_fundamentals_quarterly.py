# src/data_fetch/fetch_fundamentals_quarterly.py
"""
Fetch quarterly fundamentals from SEC EDGAR with proper point-in-time handling.

Refactored:
- per-CIK disk cache of SEC companyfacts JSON
- controlled parallel fetch (ThreadPoolExecutor) with polite rate-limiting
- robust retries/exponential backoff
- FX rate caching (disk)
- preserves public API & function signatures
"""
import os
import json
import time
import math
import requests
import yfinance as yf
import pandas as pd
import numpy as np

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------------------------
# IMPORTANT: keep this set to your name/email per SEC policy
SEC_USER_AGENT = "AnthonySacco amsacco97@gmail.com"

SEC_BASE_URL = "https://data.sec.gov"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_RATE_LIMIT_DELAY = 0.2  # recommended default (10 req/s -> 0.1s); use conservative 0.2s

# SEC filing form types to include (domestic + foreign filers)
SEC_FORM_FILTER = ['10-Q', '10-K', '10-Q/A', '10-K/A', '20-F', '20-F/A', '6-K', '6-K/A']
SEC_MAX_REPORT_LAG_DAYS = 200

# Keep your XBRL_TAG_MAP, INSTANT_FIELDS, DURATION_FIELDS, MONETARY_COLUMNS
# (omitted here for brevity in this comment — they are identical to your original map)
# For the actual file, paste the same XBRL_TAG_MAP, INSTANT_FIELDS, DURATION_FIELDS, MONETARY_COLUMNS
# I will include them verbatim below to preserve behavior.
# ---------------------------------------------------------------------------

# ----- Paste original XBRL_TAG_MAP, INSTANT_FIELDS, DURATION_FIELDS, MONETARY_COLUMNS here -----
# (for concision in this message I will reuse the exact maps from your original file)
XBRL_TAG_MAP = {
    # ... (same mapping as original) ...
    'revenue': [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'SalesRevenueNet',
        'SalesRevenueGoodsNet',
        'Revenue',
        'RevenueFromContractsWithCustomers',
        'RevenueFromSaleOfGoods',
    ],
    'net_income': [
        'NetIncomeLoss',
        'ProfitLoss',
        'ProfitLossAttributableToOwnersOfParent',
    ],
    'operating_income': [
        'OperatingIncomeLoss',
        'ProfitLossFromOperatingActivities',
    ],
    'gross_profit': ['GrossProfit'],
    'income_tax': [
        'IncomeTaxExpenseBenefit',
        'IncomeTaxExpenseContinuingOperations',
    ],
    'interest_expense': [
        'InterestExpense',
        'InterestExpenseDebt',
        'InterestAndDebtExpense',
        'FinanceCosts',
        'InterestExpenseOnBorrowings',
    ],
    'depreciation': [
        'DepreciationDepletionAndAmortization',
        'DepreciationAndAmortization',
        'Depreciation',
        'DepreciationAndAmortisationExpense',
        'AdjustmentsForDepreciationAndAmortisationExpense',
    ],
    'operating_cash_flow': [
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
        'CashFlowsFromUsedInOperatingActivities',
    ],
    'capex': [
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'CapitalExpendituresIncurredButNotYetPaid',
        'PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities',
    ],
    'sga_expense': [
        'SellingGeneralAndAdministrativeExpense',
        'AdministrativeExpense',
    ],
    'research_and_development': ['ResearchAndDevelopmentExpense'],
    'employees': ['EntityNumberOfEmployees'],
    'total_assets': ['Assets'],
    'total_equity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'EquityAttributableToOwnersOfParent',
        'Equity',
    ],
    'long_term_debt': [
        'LongTermDebt',
        'LongTermDebtNoncurrent',
        'LongTermDebtAndCapitalLeaseObligations',
        'NoncurrentBorrowings',
        'Borrowings',
    ],
    'short_term_debt': [
        'ShortTermBorrowings',
        'DebtCurrent',
        'LongTermDebtCurrent',
        'CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings',
    ],
    'total_cash': [
        'CashAndCashEquivalentsAtCarryingValue',
        'CashCashEquivalentsAndShortTermInvestments',
        'CashAndCashEquivalents',
    ],
    'current_assets': ['AssetsCurrent', 'CurrentAssets'],
    'current_liabilities': ['LiabilitiesCurrent', 'CurrentLiabilities'],
    'inventory': ['InventoryNet', 'Inventories'],
    'shares_outstanding': [
        'EntityCommonStockSharesOutstanding',
        'CommonStockSharesOutstanding',
        'NumberOfSharesOutstanding',
        'NumberOfSharesIssued',
    ],
    'receivables': [
        'AccountsReceivableNetCurrent',
        'ReceivablesNetCurrent',
        'CurrentTradeReceivables',
    ],
    'ppe': ['PropertyPlantAndEquipmentNet', 'PropertyPlantAndEquipment'],
    'real_estate_assets': [
        'RealEstateNet',
        'RealEstateInvestmentPropertyNet',
        'RealEstateGrossAtCarryingValue',
        'RealEstateInvestments',
        'InvestmentProperty',
    ],
    'secured_debt': ['DebtInstrumentCollateralAmount', 'SecuredDebt'],
    'convertible_debt': ['ConvertibleDebt'],
    'taxes_payable': ['TaxesPayableCurrent'],
}

INSTANT_FIELDS = {
    'total_assets', 'total_equity', 'long_term_debt', 'short_term_debt',
    'total_cash', 'current_assets', 'current_liabilities', 'inventory',
    'shares_outstanding', 'receivables', 'ppe', 'real_estate_assets',
    'secured_debt', 'convertible_debt', 'taxes_payable', 'employees',
}
DURATION_FIELDS = {
    'revenue', 'net_income', 'operating_income', 'gross_profit',
    'income_tax', 'interest_expense', 'depreciation', 'operating_cash_flow',
    'capex', 'sga_expense', 'research_and_development',
}
MONETARY_COLUMNS = {
    'revenue', 'net_income', 'operating_income', 'gross_profit', 'ebitda',
    'total_assets', 'total_equity', 'total_debt', 'total_cash',
    'current_assets', 'current_liabilities', 'operating_cash_flow',
    'capex', 'sga_expense', 'research_and_development', 'inventory',
    'receivables', 'ppe', 'real_estate_assets', 'secured_debt',
    'convertible_debt', 'taxes_payable', 'interest_expense',
    'depreciation', 'income_tax', 'long_term_debt', 'short_term_debt',
}
# ---------------------------------------------------------------------------

# Local caches on disk
BASE_CACHE_DIR = Path(os.path.dirname(__file__)).parent.parent / "data" / "cache"
CIK_CACHE_PATH = BASE_CACHE_DIR / "sec_cik_mapping.json"
COMPANYFACTS_CACHE_DIR = BASE_CACHE_DIR / "sec_companyfacts"
SUBMISSIONS_CACHE_DIR = BASE_CACHE_DIR / "sec_submissions"
FX_CACHE_DIR = BASE_CACHE_DIR / "fx_rates"

# Ensure dirs
COMPANYFACTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
BASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory caches
_CIK_CACHE: Dict[str, str] = {}
_FX_RATE_CACHE: Dict[str, pd.Series] = {}

# Global throttling control (shared)
_last_request_time = 0.0


# -------------------- Helpers: CIK mapping --------------------
def _load_cik_mapping() -> Dict[str, str]:
    global _CIK_CACHE
    if _CIK_CACHE:
        return _CIK_CACHE

    if CIK_CACHE_PATH.exists():
        age_days = (time.time() - CIK_CACHE_PATH.stat().st_mtime) / 86400.0
        if age_days < 30:
            with open(CIK_CACHE_PATH, "r") as f:
                _CIK_CACHE = json.load(f)
                return _CIK_CACHE

    headers = {"User-Agent": SEC_USER_AGENT}
    resp = requests.get(SEC_COMPANY_TICKERS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    mapping = {}
    for v in data.values():
        ticker = v.get("ticker", "").upper()
        cik = str(v.get("cik_str", "")).zfill(10)
        if ticker and cik:
            mapping[ticker] = cik

    # persist
    CIK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CIK_CACHE_PATH, "w") as f:
        json.dump(mapping, f)
    _CIK_CACHE = mapping
    return mapping


# -------------------- Helpers: polite SEC fetch with cache --------------------
def _polite_get(url: str, headers: dict, timeout: int = 30, max_retries: int = 3):
    """GET with exponential backoff, updates global _last_request_time."""
    global _last_request_time
    attempt = 0
    while attempt <= max_retries:
        elapsed = time.time() - _last_request_time
        if elapsed < SEC_RATE_LIMIT_DELAY:
            time.sleep(SEC_RATE_LIMIT_DELAY - elapsed)
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            _last_request_time = time.time()
            if resp.status_code == 429:
                # Backoff
                wait = 2 ** attempt
                time.sleep(wait)
                attempt += 1
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt >= max_retries:
                raise
            time.sleep(2 ** attempt)
            attempt += 1
    return None


def _fetch_company_facts_cached(cik: str, refresh_cache: bool = False) -> Optional[dict]:
    """
    Fetch companyfacts JSON from SEC, caching to disk per CIK.
    Returns parsed JSON or None.
    """
    cache_file = COMPANYFACTS_CACHE_DIR / f"CIK{cik}.json"
    if cache_file.exists() and not refresh_cache:
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            # fallback to re-fetch
            pass

    url = f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    try:
        resp = _polite_get(url, headers=headers)
        if resp is None:
            return None
        if resp.status_code == 404:
            return None
        data = resp.json()
        # Save cache
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        # bubble up None to caller
        return None


def _default_sec_metadata() -> Dict[str, Any]:
    return {"sic": np.nan, "sic2": np.nan}


def _parse_sic_codes(raw_sic: Any) -> Dict[str, Any]:
    if raw_sic is None or (isinstance(raw_sic, float) and np.isnan(raw_sic)):
        return {"sic": np.nan, "sic2": np.nan}

    digits = "".join(ch for ch in str(raw_sic) if ch.isdigit())
    if not digits:
        return {"sic": np.nan, "sic2": np.nan}

    sic = int(digits)
    sic2 = int(digits[:2]) if len(digits) >= 2 else np.nan
    return {"sic": sic, "sic2": sic2}


def _fetch_submissions_metadata_cached(cik: str, refresh_cache: bool = False) -> Dict[str, Any]:
    """
    Fetch SEC submissions metadata (SIC and description) and cache per CIK.
    """
    if not cik:
        return _default_sec_metadata()

    cache_file = SUBMISSIONS_CACHE_DIR / f"CIK{cik}.json"
    payload = None

    if cache_file.exists() and not refresh_cache:
        try:
            with open(cache_file, "r") as f:
                payload = json.load(f)
        except Exception:
            payload = None

    if payload is None:
        headers = {"User-Agent": SEC_USER_AGENT}
        url = SEC_SUBMISSIONS_URL.format(cik=cik)
        try:
            resp = _polite_get(url, headers=headers)
            if resp is not None and resp.status_code != 404:
                payload = resp.json()
                with open(cache_file, "w") as f:
                    json.dump(payload, f)
        except Exception:
            payload = None

    if not payload:
        return _default_sec_metadata()

    sic_meta = _parse_sic_codes(payload.get("sic"))
    return {
        "sic": sic_meta["sic"],
        "sic2": sic_meta["sic2"],
    }


def _fetch_sec_payloads_cached(cik: str, refresh_cache: bool = False) -> Dict[str, Any]:
    """
    Fetch all SEC payloads needed by this pipeline for one CIK.
    """
    return {
        "facts": _fetch_company_facts_cached(cik, refresh_cache=refresh_cache),
        "meta": _fetch_submissions_metadata_cached(cik, refresh_cache=refresh_cache),
    }


# -------------------- XBRL extraction (mostly preserved) --------------------
def _select_unit_entries(units: Dict[str, Any], requested_unit: str):
    """
    Pick entries for the requested unit without crossing unit families.

    SEC companyfacts may expose several units for one concept. A monetary
    request should never fall back to shares-per-unit data, and vice versa.
    """
    if not units:
        return None, None

    entries = units.get(requested_unit)
    if entries:
        return requested_unit, entries

    if requested_unit == "USD":
        for unit_key, unit_entries in units.items():
            unit_l = str(unit_key).lower()
            if unit_l in {"shares", "pure"} or "shares" in unit_l:
                continue
            if unit_entries:
                return unit_key, unit_entries
    elif requested_unit == "shares":
        entries = units.get("shares")
        if entries:
            return "shares", entries
    elif requested_unit == "pure":
        entries = units.get("pure")
        if entries:
            return "pure", entries

    return None, None


def _dedupe_prefer_original_earliest(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """
    Deduplicate SEC fact rows for point-in-time use.

    Companyfacts repeats old comparative periods in later filings. Keeping the
    latest duplicate pushes old quarters into the future. Prefer the original
    non-amended filing, then the earliest filed date, preserving XBRL tag order.
    """
    if df.empty:
        return df

    work = df.copy()
    form = work.get("form", pd.Series("", index=work.index)).fillna("").astype(str)
    work["_amended_rank"] = form.str.endswith("/A").astype(int)
    if "_tag_rank" not in work.columns:
        work["_tag_rank"] = 0
    if "_duration_rank" not in work.columns:
        work["_duration_rank"] = 0

    sort_cols = ["_amended_rank", "filed", "_duration_rank", "_tag_rank", "accn"]
    sort_cols = [c for c in sort_cols if c in work.columns]
    work = work.sort_values(sort_cols, kind="mergesort")
    work = work.drop_duplicates(subset=subset, keep="first")
    return work.drop(columns=["_amended_rank", "_duration_rank"], errors="ignore").reset_index(drop=True)


def _choose_report_date(quarter_end, filing_dates: List[Any]):
    """Use the first valid filing on/after quarter end as the quarter availability date."""
    qe = pd.to_datetime(quarter_end, errors="coerce")
    candidates = pd.to_datetime(pd.Series(filing_dates), errors="coerce").dropna()
    if candidates.empty:
        return pd.NaT
    if pd.notna(qe):
        after_qe = candidates[candidates >= qe]
        if not after_qe.empty:
            return after_qe.min()
    return candidates.min()


def _filter_quarter_date_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove quarter rows that cannot be aligned point-in-time.

    Very old companyfacts rows sometimes only appear as later comparative
    periods, not as their original filing. Keeping those rows creates stale
    quarter/report-date pairings, so exclude them from production output.
    """
    if df.empty:
        return df

    quarter_end = pd.to_datetime(df["quarter_end_date"], errors="coerce")
    report_date = pd.to_datetime(df["report_date"], errors="coerce")
    lag_days = (report_date - quarter_end).dt.days

    valid = (
        quarter_end.notna()
        & report_date.notna()
        & (lag_days >= 0)
        & (lag_days <= SEC_MAX_REPORT_LAG_DAYS)
    )
    return df.loc[valid].reset_index(drop=True)


def _extract_fact_series(
    facts_data: dict,
    xbrl_tags: List[str],
    unit: str = "USD",
    form_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Union entries from all matching tags and deduplicate by period for PIT use.
    Returns columns: [val, end, start (opt), filed, form, fp, fy, accn]
    """
    if not facts_data or "facts" not in facts_data:
        return pd.DataFrame()

    facts = facts_data.get("facts", {})
    namespaces = [facts.get("us-gaap", {}), facts.get("ifrs-full", {}), facts.get("dei", {})]
    tag_rank = {tag: rank for rank, tag in enumerate(xbrl_tags)}

    rows = []
    for tag in xbrl_tags:
        tag_obj = None
        ns_name = None
        for ns in namespaces:
            if tag in ns:
                tag_obj = ns[tag]
                break
        if not tag_obj:
            continue

        units = tag_obj.get("units", {})
        unit_key, entries = _select_unit_entries(units, unit)
        if not entries:
            continue

        for e in entries:
            # keep only relevant keys; leave extras untouched
            rows.append({
                "val": e.get("val"),
                "end": e.get("end"),
                "start": e.get("start"),
                "filed": e.get("filed"),
                "form": e.get("form"),
                "fp": e.get("fp"),
                "fy": e.get("fy"),
                "accn": e.get("accn"),
                "frame": e.get("frame"),
                "unit": unit_key,
                "_tag": tag,
                "_tag_rank": tag_rank.get(tag, len(tag_rank)),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # parse datetimes
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"], errors="coerce").astype("datetime64[ns]")
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"], errors="coerce").astype("datetime64[ns]")
    if "filed" in df.columns:
        df["filed"] = pd.to_datetime(df["filed"], errors="coerce").astype("datetime64[ns]")

    if form_filter and "form" in df.columns:
        df = df[df["form"].isin(form_filter)].copy()

    # Deduplicate by period. SEC companyfacts repeats old comparative values in
    # newer filings; for PIT alignment, keep the first original filing.
    dedup_cols = ["end"]
    if "start" in df.columns and df["start"].notna().any():
        dedup_cols.append("start")

    df = _dedupe_prefer_original_earliest(df, dedup_cols)
    return df


def _safe_div(numerator, denominator):
    try:
        if (pd.isna(numerator) or pd.isna(denominator) or denominator == 0):
            return np.nan
        return numerator / denominator
    except Exception:
        return np.nan


# The _extract_quarterly_instant and _extract_quarterly_duration logic remains similar to original
# but uses the slightly optimized _extract_fact_series above. We keep original semantics.
def _extract_quarterly_instant(facts_data: dict, field_name: str) -> pd.DataFrame:
    unit = "shares" if field_name in {"shares_outstanding"} else ("pure" if field_name in {"employees"} else "USD")
    tags = XBRL_TAG_MAP.get(field_name, [])
    raw = _extract_fact_series(facts_data, tags, unit=unit, form_filter=SEC_FORM_FILTER)
    if raw.empty:
        return pd.DataFrame(columns=["quarter_end", "value", "filed"])
    raw = _dedupe_prefer_original_earliest(raw, ["end"])
    return pd.DataFrame({
        "quarter_end": raw["end"].values,
        "value": raw["val"].values,
        "filed": raw["filed"].values,
    })


def _extract_quarterly_duration(facts_data: dict, field_name: str) -> pd.DataFrame:
    unit = "USD"
    tags = XBRL_TAG_MAP.get(field_name, [])
    raw = _extract_fact_series(facts_data, tags, unit=unit, form_filter=SEC_FORM_FILTER)
    if raw.empty:
        return pd.DataFrame(columns=["quarter_end", "value", "filed"])

    results = {}

    if "start" in raw.columns:
        qdata = raw.dropna(subset=["start", "end"]).copy()
        if not qdata.empty:
            qdata["duration_days"] = (qdata["end"] - qdata["start"]).dt.days

            # Step 1: single-quarter entries
            single_q = qdata[(qdata["duration_days"] >= 60) & (qdata["duration_days"] <= 120)].copy()
            single_q["_duration_rank"] = (single_q["duration_days"] - 90).abs()
            single_q = _dedupe_prefer_original_earliest(single_q, ["end"])
            for _, row in single_q.iterrows():
                results[row["end"]] = (row["val"], row["filed"])

            # Step 2: fill gaps from cumulative YTD entries by differencing
            cumulative = qdata[(qdata["duration_days"] > 120) & (qdata["duration_days"] < 400)].copy()
            if not cumulative.empty:
                cumulative = _dedupe_prefer_original_earliest(cumulative, ["start", "end"])
                # Group by fiscal-year START date
                for fy_start in cumulative["start"].dropna().unique():
                    fy_cums = cumulative[cumulative["start"] == fy_start].sort_values("end")
                    prev_val = None
                    # See if we have a Q1 single-quarter value. If not, do not
                    # treat the first YTD cumulative value as a single quarter.
                    for qe, (qv, _) in list(results.items()):
                        if fy_start <= qe <= fy_cums["end"].max():
                            days_from_start = (qe - fy_start).days
                            if 60 <= days_from_start <= 120:
                                prev_val = qv
                                break
                    for _, crow in fy_cums.iterrows():
                        if crow["end"] in results:
                            prev_val = crow["val"]
                            continue
                        if prev_val is None:
                            prev_val = crow["val"]
                            continue
                        q_val = crow["val"] - prev_val
                        results[crow["end"]] = (q_val, crow["filed"])
                        prev_val = crow["val"]

    # Step 3: derive Q4 from annual reports
    if "start" in raw.columns:
        annual = raw.dropna(subset=["start", "end"]).copy()
        if not annual.empty:
            annual["duration_days"] = (annual["end"] - annual["start"]).dt.days
            fy_rows = annual[annual["duration_days"] >= 300]
        else:
            fy_rows = pd.DataFrame()
    else:
        fy_rows = pd.DataFrame()

    if fy_rows.empty and "fp" in raw.columns:
        annual = raw[raw["form"].isin(["10-K", "10-K/A"])]
        fy_rows = annual[annual["fp"] == "FY"]

    if not fy_rows.empty:
        fy_rows = _dedupe_prefer_original_earliest(fy_rows, ["start", "end"])
        for _, fy_row in fy_rows.iterrows():
            fy_end = fy_row["end"]
            fy_start = fy_row.get("start", pd.NaT)
            fy_val = fy_row["val"]
            fy_filed = fy_row["filed"]
            if fy_end in results:
                continue
            q_sum = 0.0
            q_count = 0
            for qe, (qv, _) in results.items():
                if pd.notna(fy_start) and not (fy_start <= qe < fy_end):
                    continue
                days_before = (fy_end - qe).days
                if 0 < days_before <= 300:
                    q_sum += qv
                    q_count += 1
            if q_count >= 2:
                q4_val = fy_val - q_sum
                results[fy_end] = (q4_val, fy_filed)

    if not results:
        return pd.DataFrame(columns=["quarter_end", "value", "filed"])

    result_df = pd.DataFrame([{"quarter_end": k, "value": v[0], "filed": v[1]} for k, v in results.items()])
    return result_df


def _snap_to_nearest(target_date, reference_dates, max_days=45):
    best = None
    best_dist = max_days + 1
    for ref in reference_dates:
        dist = abs((target_date - ref).days)
        if dist < best_dist:
            best_dist = dist
            best = ref
    return best


def _detect_reporting_currency(facts_data: dict) -> str:
    facts = facts_data.get("facts", {})
    for ns_name in ["us-gaap", "ifrs-full"]:
        ns = facts.get(ns_name, {})
        for probe_tag in ["Assets", "Revenue", "Revenues", "ProfitLoss", "NetIncomeLoss"]:
            if probe_tag in ns:
                units = ns[probe_tag].get("units", {})
                if "USD" in units:
                    return "USD"
                for u in units:
                    if u not in ("shares", "pure", "USD/shares"):
                        return u
    return "USD"


# -------------------- FX helper (persistent cache) --------------------
def _get_fx_rates(currency: str, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Convert currency -> USD using yfinance. Cache per currency on disk.
    Returns a Series indexed by `dates` with conversion rates (multiplier).
    """
    if currency == "USD" or not currency:
        return pd.Series(1.0, index=dates)

    cache_file = FX_CACHE_DIR / f"{currency}.parquet"
    # if in-memory cache exists and covers dates, use it
    if currency in _FX_RATE_CACHE:
        cached = _FX_RATE_CACHE[currency]
    else:
        pair = f"{currency}USD=X"
        try:
            # fetch a little wider window
            start = (dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            end = (dates.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            fx_df = yf.download(pair, start=start, end=end, progress=False)
            if fx_df.empty:
                _FX_RATE_CACHE[currency] = pd.Series(dtype=float)
                cached = _FX_RATE_CACHE[currency]
            else:
                close = fx_df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                # persist to disk
                close.index = pd.to_datetime(close.index).astype("datetime64[ns]")
                close.to_frame(name="rate").to_parquet(cache_file)
                cached = close
                _FX_RATE_CACHE[currency] = cached
        except Exception:
            _FX_RATE_CACHE[currency] = pd.Series(dtype=float)
            cached = _FX_RATE_CACHE[currency]

    if isinstance(cached, pd.Series) and not cached.empty:
        combined = cached.reindex(cached.index.union(dates)).sort_index().ffill().bfill()
        return combined.reindex(dates).fillna(method="ffill").fillna(method="bfill").fillna(1.0)
    else:
        # fallback to 1.0 if no FX data
        return pd.Series(1.0, index=dates)


# -------------------- Build quarterly DataFrame --------------------
def _build_quarterly_dataframe(facts_data: dict) -> Optional[pd.DataFrame]:
    duration_series = {}
    instant_series = {}

    for field in DURATION_FIELDS:
        series = _extract_quarterly_duration(facts_data, field)
        if not series.empty:
            duration_series[field] = series

    for field in INSTANT_FIELDS:
        series = _extract_quarterly_instant(facts_data, field)
        if not series.empty:
            instant_series[field] = series

    if not duration_series and not instant_series:
        return None

    # master quarters from duration fields
    master_quarters = set()
    filing_dates = {}
    for field, series in duration_series.items():
        for _, row in series.iterrows():
            qe = row["quarter_end"]
            if pd.isna(qe):
                continue
            master_quarters.add(qe)
            filing_dates.setdefault(qe, []).append(row["filed"])

    # fallback to instant fields if no duration
    if not master_quarters:
        for field, series in instant_series.items():
            for _, row in series.iterrows():
                qe = row["quarter_end"]
                if pd.isna(qe):
                    continue
                if qe.day >= 25 or qe.day <= 5:
                    master_quarters.add(qe)
                    filing_dates.setdefault(qe, []).append(row["filed"])

    if not master_quarters:
        return None

    sorted_quarters = sorted(master_quarters)

    # collect filing dates from instant fields
    for field, series in instant_series.items():
        for _, row in series.iterrows():
            snapped = _snap_to_nearest(row["quarter_end"], sorted_quarters, max_days=45)
            if snapped is not None:
                filing_dates.setdefault(snapped, []).append(row["filed"])

    rows = []
    for qe in sorted_quarters:
        row = {"quarter_end_date": qe, "report_date": _choose_report_date(qe, filing_dates.get(qe, []))}
        for field, series in duration_series.items():
            match = series[series["quarter_end"] == qe]
            row[field] = (match.iloc[0]["value"] if not match.empty else np.nan)
        for field, series in instant_series.items():
            sc = series.copy()
            sc["_dist"] = (sc["quarter_end"] - qe).abs().dt.days
            close = sc[sc["_dist"] <= 45]
            row[field] = (close.sort_values("_dist").iloc[0]["value"] if not close.empty else np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _filter_quarter_date_quality(df)
    if df.empty:
        return None

    # ensure all expected fields exist
    expected_fields = sorted(list(INSTANT_FIELDS | DURATION_FIELDS))
    for f in expected_fields:
        if f not in df.columns:
            df[f] = np.nan

    # aggregate total_debt and compute ebitda & ratios
    lt = df.get("long_term_debt", pd.Series(np.nan, index=df.index))
    st = df.get("short_term_debt", pd.Series(np.nan, index=df.index))
    df["total_debt"] = lt.fillna(0) + st.fillna(0)
    df.loc[lt.isna() & st.isna(), "total_debt"] = np.nan
    df = df.drop(columns=["long_term_debt", "short_term_debt"], errors="ignore")

    ebitda_components = ["net_income", "income_tax", "interest_expense", "depreciation"]
    if all(c in df.columns for c in ebitda_components):
        df["ebitda"] = df[ebitda_components].sum(axis=1, min_count=1)
    else:
        df["ebitda"] = np.nan

    df["profit_margin"] = df.apply(lambda r: _safe_div(r.get("net_income"), r.get("revenue")), axis=1)
    df["operating_margin"] = df.apply(lambda r: _safe_div(r.get("operating_income"), r.get("revenue")), axis=1)
    df["gross_margin"] = df.apply(lambda r: _safe_div(r.get("gross_profit"), r.get("revenue")), axis=1)
    df["roe"] = df.apply(lambda r: _safe_div(r.get("net_income"), r.get("total_equity")), axis=1)
    df["roa"] = df.apply(lambda r: _safe_div(r.get("net_income"), r.get("total_assets")), axis=1)
    df["debt_to_equity"] = df.apply(lambda r: _safe_div(r.get("total_debt"), r.get("total_equity")), axis=1)
    df["current_ratio"] = df.apply(lambda r: _safe_div(r.get("current_assets"), r.get("current_liabilities")), axis=1)
    ca = df.get("current_assets", pd.Series(np.nan, index=df.index))
    inv = df.get("inventory", pd.Series(np.nan, index=df.index))
    cl = df.get("current_liabilities", pd.Series(np.nan, index=df.index))
    df["quick_ratio"] = (ca - inv.fillna(0)) / cl.replace(0, np.nan)
    df["book_value"] = df.get("total_equity")
    df["asset_turnover"] = df.apply(lambda r: _safe_div(r.get("revenue"), r.get("total_assets")), axis=1)

    # FX conversion if needed
    currency = _detect_reporting_currency(facts_data)
    if currency != "USD":
        dates = pd.DatetimeIndex(df["quarter_end_date"].values)
        fx = _get_fx_rates(currency, dates)
        for col in MONETARY_COLUMNS:
            if col in df.columns and df[col].notna().any():
                try:
                    df[col] = df[col].astype(float) * fx.values
                except Exception:
                    pass
        for col in ["total_debt", "ebitda", "book_value"]:
            if col in df.columns and df[col].notna().any():
                try:
                    df[col] = df[col].astype(float) * fx.values
                except Exception:
                    pass

    return df


# -------------------- Public API (preserved signatures) --------------------
def fetch_quarterly_fundamentals(ticker: str, reporting_lag_days: int = 45, refresh_cache: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch quarterly fundamentals for a single ticker (preserves original signature).
    reporting_lag_days kept for compatibility but actual SEC filing dates are used.

    refresh_cache: if True, re-download companyfacts JSON for this ticker.
    """
    try:
        cik_map = _load_cik_mapping()
        cik = cik_map.get(ticker.upper())
        if cik is None:
            print(f"  No CIK found for {ticker}")
            return None

        facts = _fetch_company_facts_cached(cik, refresh_cache=refresh_cache)
        sec_meta = _fetch_submissions_metadata_cached(cik, refresh_cache=refresh_cache)
        if facts is None:
            print(f"  No EDGAR data for {ticker} (CIK {cik})")
            return None

        df = _build_quarterly_dataframe(facts)
        if df is None or df.empty:
            print(f"  No quarterly data extracted for {ticker}")
            return None

        df["ticker"] = ticker.upper()
        df["sic"] = sec_meta.get("sic", np.nan)
        df["sic2"] = sec_meta.get("sic2", np.nan)
        df = df.sort_values("quarter_end_date").reset_index(drop=True)

        # growth metrics
        revenue = df["revenue"] if "revenue" in df.columns else pd.Series(np.nan, index=df.index)
        net_income = df["net_income"] if "net_income" in df.columns else pd.Series(np.nan, index=df.index)
        df["revenue_growth_qoq"] = revenue.pct_change()
        df["earnings_growth_qoq"] = net_income.pct_change()
        df["revenue_growth_yoy"] = revenue.pct_change(periods=4)
        df["earnings_growth_yoy"] = net_income.pct_change(periods=4)

        # format dates
        df["quarter_end_date"] = pd.to_datetime(df["quarter_end_date"]).dt.strftime("%Y-%m-%d")
        df["report_date"] = pd.to_datetime(df["report_date"]).dt.strftime("%Y-%m-%d")

        return df
    except Exception as e:
        print(f"  Error fetching quarterly fundamentals for {ticker}: {e}")
        return None


def fetch_current_valuation_metrics(ticker: str) -> Dict:
    """
    Same signature as before; small safety around yfinance.info call.
    """
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        return {
            "ticker": ticker,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "beta": info.get("beta"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "float_shares": info.get("floatShares"),
            "held_percent_insiders": info.get("heldPercentInsiders"),
            "held_percent_institutions": info.get("heldPercentInstitutions"),
            "dividend_yield": info.get("dividendYield"),
            "book_value": info.get("bookValue"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "trailing_pe": info.get("trailingPE"),
        }
    except Exception as e:
        print(f"  Error fetching valuation for {ticker}: {e}")
        return None


def fetch_all_quarterly_fundamentals(
    tickers: List[str],
    save_dir: str = "data/raw/fundamentals_quarterly/",
    reporting_lag_days: int = 45,
    max_workers: int = 4,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch quarterly fundamentals for a list of tickers. Uses controlled parallelism,
    per-CIK caching, and returns a combined DataFrame (and writes parquet to save_dir).

    max_workers: number of concurrent threads (keep small to respect SEC)
    refresh_cache: if True forces re-download of companyfacts JSONs
    """
    os.makedirs(save_dir, exist_ok=True)
    _load_cik_mapping()

    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    all_frames = []
    failed = []
    temp_out_dir = Path(save_dir) / "per_ticker"
    temp_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching quarterly fundamentals from SEC EDGAR for {len(tickers)} tickers...")
    print(f"Using actual SEC filing dates (no estimated lag). Workers={max_workers}")

    fetch_partial = partial(_fetch_sec_payloads_cached, refresh_cache=refresh_cache)
    cik_map = _load_cik_mapping()

    # Fetch SEC payloads (companyfacts + submissions metadata) in parallel, then
    # process sequentially to build per-ticker DataFrames.
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {}
        for tick in tickers:
            cik = cik_map.get(tick)
            if not cik:
                failed.append(tick)
                continue
            futures[exe.submit(fetch_partial, cik)] = (tick, cik)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="EDGAR fetch"):
            tick, cik = futures[fut]
            try:
                payload = fut.result()
                facts = (payload or {}).get("facts")
                sec_meta = (payload or {}).get("meta") or _default_sec_metadata()
                if facts is None:
                    failed.append(tick)
                    continue
                # Build DF for this ticker
                try:
                    df = _build_quarterly_dataframe(facts)
                    if df is None or df.empty:
                        failed.append(tick)
                        continue
                    df["ticker"] = tick
                    df["sic"] = sec_meta.get("sic", np.nan)
                    df["sic2"] = sec_meta.get("sic2", np.nan)
                    df = df.sort_values("quarter_end_date").reset_index(drop=True)
                    # compute growth columns (same as single-ticker function)
                    revenue = df["revenue"] if "revenue" in df.columns else pd.Series(np.nan, index=df.index)
                    net_income = df["net_income"] if "net_income" in df.columns else pd.Series(np.nan, index=df.index)
                    df["revenue_growth_qoq"] = revenue.pct_change()
                    df["earnings_growth_qoq"] = net_income.pct_change()
                    df["revenue_growth_yoy"] = revenue.pct_change(periods=4)
                    df["earnings_growth_yoy"] = net_income.pct_change(periods=4)
                    # format dates before saving
                    df["quarter_end_date"] = pd.to_datetime(df["quarter_end_date"]).dt.strftime("%Y-%m-%d")
                    df["report_date"] = pd.to_datetime(df["report_date"]).dt.strftime("%Y-%m-%d")
                    # persist per-ticker partial
                    df.to_parquet(temp_out_dir / f"{tick}.parquet", index=False)
                    all_frames.append(df)
                except Exception:
                    failed.append(tick)
            except Exception:
                failed.append(tick)

    if not all_frames:
        print("No data fetched!")
        return pd.DataFrame()

    fundamentals_df = pd.concat(all_frames, ignore_index=True)
    output_path = os.path.join(save_dir, "quarterly_fundamentals.parquet")
    fundamentals_df.to_parquet(output_path, index=False)

    if failed:
        failed_path = os.path.join(save_dir, "failed_tickers.json")
        with open(failed_path, "w") as f:
            json.dump(sorted(set(failed)), f, indent=2)
        print(f"\nWarning: Failed to fetch {len(failed)} tickers (see {failed_path})")

    print("\n=== Summary ===")
    print(f"Total quarterly records: {len(fundamentals_df)}")
    print(f"Unique tickers: {fundamentals_df['ticker'].nunique()}")
    print(f"Date range: {fundamentals_df['quarter_end_date'].min()} to {fundamentals_df['quarter_end_date'].max()}")
    print(f"Saved to: {output_path}")

    return fundamentals_df


def merge_fundamentals_point_in_time(price_df: pd.DataFrame,
                                     fundamentals_df: pd.DataFrame,
                                     valuation_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Merge quarterly fundamentals with price data using point-in-time logic.
    Preserves original signature/behavior but uses robust datetime conversion.
    """
    print("Performing point-in-time merge of fundamentals...")
    price = price_df.copy()
    fund = fundamentals_df.copy()

    # enforce required columns & types
    price["date"] = pd.to_datetime(price["date"], errors="coerce").astype("datetime64[ns]")
    fund["report_date"] = pd.to_datetime(fund["report_date"], errors="coerce").astype("datetime64[ns]")
    fund["quarter_end_date"] = pd.to_datetime(fund["quarter_end_date"], errors="coerce").astype("datetime64[ns]")

    # merge_asof requires globally sorted join keys; ticker-level sort is applied after merge
    bad_price_dates = int(price["date"].isna().sum())
    if bad_price_dates:
        raise ValueError(
            f"price_df contains {bad_price_dates} invalid 'date' values after datetime parsing."
        )
    price = price.sort_values(["date", "ticker"]).reset_index(drop=True)
    fund = fund[fund["report_date"].notna()].sort_values(["report_date", "ticker"]).reset_index(drop=True)

    merged = pd.merge_asof(
        price,
        fund,
        left_on="date",
        right_on="report_date",
        by="ticker",
        direction="backward",
        suffixes=("", "_fundamental")
    )

    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    if valuation_df is not None:
        print("Warning: valuation_df provided but ignored to avoid look-ahead leakage.")

    print(f"Merged {len(merged)} rows")
    cov = merged["quarter_end_date"].notna().sum() / max(1, len(merged)) * 100
    print(f"Fundamentals coverage: {cov:.1f}%")

    return merged


def _load_tickers_from_file(path: str) -> List[str]:
    """Load tickers from JSON list/dict or CSV with a ticker column."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Universe file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            tickers = payload
        elif isinstance(payload, dict) and "tickers" in payload:
            tickers = payload["tickers"]
        else:
            raise ValueError(f"Unsupported JSON universe format in {path}")
    elif ext == ".csv":
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            raise ValueError(f"CSV universe file must include a 'ticker' column: {path}")
        tickers = df["ticker"].tolist()
    else:
        raise ValueError(f"Unsupported universe file extension: {path}")

    return [str(t).strip().upper().replace(".", "-") for t in tickers if str(t).strip()]


def _resolve_tickers(tickers_arg: Optional[str], universe_file: Optional[str]) -> List[str]:
    if universe_file:
        return _load_tickers_from_file(universe_file)
    if not tickers_arg:
        raise ValueError("One of --tickers or --universe_file is required.")
    if tickers_arg.endswith(".json") or tickers_arg.endswith(".csv"):
        return _load_tickers_from_file(tickers_arg)
    return [t.strip().upper().replace(".", "-") for t in tickers_arg.split(",") if t.strip()]


def _filter_by_date_window(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """
    Clip rows to requested date window while preserving enough history
    for point-in-time merges near the start boundary.
    """
    if df.empty:
        return df

    out = df.copy()
    out["quarter_end_date"] = pd.to_datetime(out["quarter_end_date"], errors="coerce")
    out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")

    if end_date:
        end_ts = pd.Timestamp(end_date)
        out = out[out["report_date"] <= end_ts]
    if start_date:
        # Keep one year prior to start so early-window joins still have history.
        start_with_buffer = pd.Timestamp(start_date) - pd.DateOffset(years=1)
        out = out[out["quarter_end_date"] >= start_with_buffer]

    out["quarter_end_date"] = out["quarter_end_date"].dt.strftime("%Y-%m-%d")
    out["report_date"] = out["report_date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


# If run as script, keep compatible CLI like original
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch quarterly fundamentals from SEC EDGAR (refactor)")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers OR path to JSON/CSV file")
    parser.add_argument("--universe_file", type=str, default=None,
                        help="Path to JSON/CSV universe file")
    parser.add_argument("--start_date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data_tag", type=str, default=None, help="Run tag for output folder (default: timestamp)")
    parser.add_argument("--raw_root", type=str, default="data/raw", help="Root raw data directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (keep small for SEC)")
    parser.add_argument("--refresh_cache", action="store_true", help="Refresh SEC companyfacts cache")
    args = parser.parse_args()

    data_tag = args.data_tag or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = args.save_dir or os.path.join(args.raw_root, data_tag, "fundamentals_quarterly")
    tickers = _resolve_tickers(args.tickers, args.universe_file)

    print(f"Fetching quarterly fundamentals for {len(tickers)} tickers...")
    print(f"Data tag: {data_tag}")
    print(f"Output dir: {save_dir}")

    fundamentals_df = fetch_all_quarterly_fundamentals(
        tickers, save_dir=save_dir, max_workers=args.workers, refresh_cache=args.refresh_cache
    )
    fundamentals_df = _filter_by_date_window(
        fundamentals_df,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if not fundamentals_df.empty:
        output_path = os.path.join(save_dir, "quarterly_fundamentals.parquet")
        fundamentals_df.to_parquet(output_path, index=False)
        print(f"Filtered output saved to: {output_path}")

    print("\n=== Sample Data ===")
    print(fundamentals_df.head(10))

    if not fundamentals_df.empty:
        completeness = (1 - fundamentals_df.isnull().sum() / len(fundamentals_df)) * 100
        print("\n=== Data Completeness ===")
        print(completeness.sort_values(ascending=False).head(40))
