# src/data_fetch/fetch_fundamentals_quarterly.py
"""
Fetch quarterly fundamentals from SEC EDGAR with proper point-in-time handling.

Uses the SEC EDGAR Company Facts API (XBRL) to pull full quarterly history
back to ~2009 with actual SEC filing dates (no estimated reporting lag needed).
"""
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
from typing import List, Dict, Optional
from datetime import datetime

# ---------------------------------------------------------------------------
# SEC EDGAR configuration
# ---------------------------------------------------------------------------
# IMPORTANT: Set this to your name and email before using.
# SEC requires a User-Agent header identifying the requester.
SEC_USER_AGENT = "AnthonySacco amsacco97@gmail.com"

SEC_BASE_URL = "https://data.sec.gov"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_RATE_LIMIT_DELAY = 0.2  # 10 req/s max -> ~200ms between requests

# XBRL tag mapping: internal field name -> ordered list of us-gaap tags to try
# First match wins. Companies use different tags for the same concept.
XBRL_TAG_MAP = {
    # Income statement (duration)
    'revenue': [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'SalesRevenueNet',
        'SalesRevenueGoodsNet',
    ],
    'net_income': [
        'NetIncomeLoss',
        'ProfitLoss',
    ],
    'operating_income': [
        'OperatingIncomeLoss',
    ],
    'gross_profit': [
        'GrossProfit',
    ],
    'income_tax': [
        'IncomeTaxExpenseBenefit',
    ],
    'interest_expense': [
        'InterestExpense',
        'InterestExpenseDebt',
        'InterestAndDebtExpense',
    ],
    'depreciation': [
        'DepreciationDepletionAndAmortization',
        'DepreciationAndAmortization',
        'Depreciation',
    ],
    'operating_cash_flow': [
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
    ],
    'capex': [
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'CapitalExpendituresIncurredButNotYetPaid',
    ],
    'sga_expense': [
        'SellingGeneralAndAdministrativeExpense',
    ],
    'research_and_development': [
        'ResearchAndDevelopmentExpense',
    ],
    'employees': [
        'EntityNumberOfEmployees',
    ],
    # Balance sheet (instant)
    'total_assets': [
        'Assets',
    ],
    'total_equity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    ],
    'long_term_debt': [
        'LongTermDebt',
        'LongTermDebtNoncurrent',
        'LongTermDebtAndCapitalLeaseObligations',
    ],
    'short_term_debt': [
        'ShortTermBorrowings',
        'DebtCurrent',
        'LongTermDebtCurrent',
    ],
    'total_cash': [
        'CashAndCashEquivalentsAtCarryingValue',
        'CashCashEquivalentsAndShortTermInvestments',
    ],
    'current_assets': [
        'AssetsCurrent',
    ],
    'current_liabilities': [
        'LiabilitiesCurrent',
    ],
    'inventory': [
        'InventoryNet',
        'Inventories',
    ],
    'shares_outstanding': [
        'EntityCommonStockSharesOutstanding',
        'CommonStockSharesOutstanding',
    ],
    'receivables': [
        'AccountsReceivableNetCurrent',
        'ReceivablesNetCurrent',
    ],
    'ppe': [
        'PropertyPlantAndEquipmentNet',
    ],
    'real_estate_assets': [
        'RealEstateNet',
        'RealEstateInvestmentPropertyNet',
        'RealEstateGrossAtCarryingValue',
        'RealEstateInvestments',
    ],
    'secured_debt': [
        'DebtInstrumentCollateralAmount',
        'SecuredDebt',
    ],
    'convertible_debt': [
        'ConvertibleDebt',
    ],
    'taxes_payable': [
        'TaxesPayableCurrent',
    ],
}

# Fields that are balance sheet (instant/point-in-time) vs income statement (duration)
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

# ---------------------------------------------------------------------------
# CIK mapping (cached)
# ---------------------------------------------------------------------------
_CIK_CACHE: Dict[str, str] = {}


def _load_cik_mapping() -> Dict[str, str]:
    """Load and cache the ticker-to-CIK mapping from SEC."""
    global _CIK_CACHE
    if _CIK_CACHE:
        return _CIK_CACHE

    cache_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'cache', 'sec_cik_mapping.json'
    )

    # Try loading from local cache (refresh every 30 days)
    if os.path.exists(cache_path):
        file_age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
        if file_age_days < 30:
            with open(cache_path, 'r') as f:
                _CIK_CACHE = json.load(f)
                return _CIK_CACHE

    # Fetch from SEC
    headers = {'User-Agent': SEC_USER_AGENT}
    resp = requests.get(SEC_COMPANY_TICKERS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    _CIK_CACHE = {}
    for entry in data.values():
        ticker = entry['ticker'].upper()
        cik = str(entry['cik_str']).zfill(10)
        _CIK_CACHE[ticker] = cik

    # Save to local cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(_CIK_CACHE, f)

    print(f"Loaded CIK mapping for {len(_CIK_CACHE)} tickers")
    return _CIK_CACHE


# ---------------------------------------------------------------------------
# EDGAR API helpers
# ---------------------------------------------------------------------------
_last_request_time = 0.0


def _fetch_company_facts(cik: str) -> Optional[dict]:
    """Fetch all XBRL facts for a company from SEC EDGAR with rate limiting and retries."""
    global _last_request_time

    url = f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {'User-Agent': SEC_USER_AGENT}

    for attempt in range(3):
        # Rate limiting
        elapsed = time.time() - _last_request_time
        if elapsed < SEC_RATE_LIMIT_DELAY:
            time.sleep(SEC_RATE_LIMIT_DELAY - elapsed)

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            _last_request_time = time.time()

            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise

    return None


def _extract_fact_series(
    facts_data: dict,
    xbrl_tags: List[str],
    unit: str = 'USD',
    form_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract a time series from EDGAR company facts by unioning ALL matching XBRL tags.

    Companies switch tags over time (e.g., SalesRevenueNet -> RevenueFromContract
    after ASC 606 adoption). Picking only one tag loses data. Instead, we union
    entries from all matching tags and deduplicate by (end, start) keeping the
    latest filing.

    Returns DataFrame with columns: [val, end, start, filed, form, fp, fy, accn]
    """
    facts = facts_data.get('facts', {})
    namespaces = [facts.get('us-gaap', {}), facts.get('dei', {})]

    all_dfs = []

    for tag in xbrl_tags:
        tag_obj = None
        for namespace in namespaces:
            if tag in namespace:
                tag_obj = namespace[tag]
                break
        if tag_obj is None:
            continue

        units = tag_obj.get('units', {})
        entries = units.get(unit) or units.get('shares') or units.get('USD/shares')
        if not entries:
            continue

        df = pd.DataFrame(entries)
        if df.empty:
            continue

        # Filter by form type
        if form_filter and 'form' in df.columns:
            df = df[df['form'].isin(form_filter)]

        if df.empty:
            continue

        # Parse dates
        df['end'] = pd.to_datetime(df['end'], errors='coerce')
        df['filed'] = pd.to_datetime(df['filed'], errors='coerce')
        if 'start' in df.columns:
            df['start'] = pd.to_datetime(df['start'], errors='coerce')

        df['_tag'] = tag
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate: for the same (end, start) period, keep the latest filing.
    # This handles both tag-switching and amended filings.
    dedup_cols = ['end']
    if 'start' in combined.columns:
        dedup_cols.append('start')
    combined = combined.sort_values('filed').drop_duplicates(subset=dedup_cols, keep='last')
    combined = combined.drop(columns=['_tag'], errors='ignore')

    return combined


def _safe_div(numerator, denominator):
    """Safe division returning NaN when denominator is 0 or NaN."""
    try:
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return np.nan
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return np.nan


# ---------------------------------------------------------------------------
# Core quarterly extraction logic
# ---------------------------------------------------------------------------

def _extract_quarterly_instant(
    facts_data: dict, field_name: str
) -> pd.DataFrame:
    """
    Extract quarterly balance sheet values (instant/point-in-time).

    Returns DataFrame with columns: [quarter_end, value, filed]
    """
    unit = 'shares' if field_name in {'shares_outstanding'} else ('pure' if field_name in {'employees'} else 'USD')
    tags = XBRL_TAG_MAP.get(field_name, [])
    raw = _extract_fact_series(
        facts_data, tags, unit=unit,
        form_filter=['10-Q', '10-K', '10-Q/A', '10-K/A'],
    )
    if raw.empty:
        return pd.DataFrame(columns=['quarter_end', 'value', 'filed'])

    # For instant values, the 'end' date IS the measurement date
    # Deduplicate: keep the latest filed per end date
    raw = raw.sort_values('filed').drop_duplicates(subset=['end'], keep='last')

    result = pd.DataFrame({
        'quarter_end': raw['end'].values,
        'value': raw['val'].values,
        'filed': raw['filed'].values,
    })
    return result


def _extract_quarterly_duration(
    facts_data: dict, field_name: str
) -> pd.DataFrame:
    """
    Extract quarterly income statement / cash-flow values (single-quarter duration).

    Handles:
    - Single-quarter entries from 10-Q (~90-day duration)
    - Gap-filling from cumulative YTD entries via differencing
      (cash-flow statements in 10-Q typically report YTD, not single-quarter)
    - Q4 derivation from 10-K annual minus sum(Q1+Q2+Q3)

    Returns DataFrame with columns: [quarter_end, value, filed]
    """
    unit = 'USD'
    tags = XBRL_TAG_MAP.get(field_name, [])
    raw = _extract_fact_series(
        facts_data, tags, unit=unit,
        form_filter=['10-Q', '10-K', '10-Q/A', '10-K/A'],
    )
    if raw.empty:
        return pd.DataFrame(columns=['quarter_end', 'value', 'filed'])

    results = {}  # quarter_end -> (value, filed)

    if 'start' in raw.columns:
        q_data = raw.dropna(subset=['start', 'end']).copy()
        if not q_data.empty:
            q_data['duration_days'] = (q_data['end'] - q_data['start']).dt.days

            # --- Step 1: Single-quarter entries (60-120 days) ---
            single_q = q_data[(q_data['duration_days'] >= 60) & (q_data['duration_days'] <= 120)].copy()
            single_q = single_q.sort_values('filed').drop_duplicates(subset=['end'], keep='last')
            for _, row in single_q.iterrows():
                results[row['end']] = (row['val'], row['filed'])

            # --- Step 2: Fill gaps from cumulative YTD entries ---
            # Cash-flow statements often only have YTD cumulative in 10-Qs.
            # E.g., Q1=90d (single), Q2=181d (cumulative), Q3=273d (cumulative).
            # Derive single-quarter values by differencing consecutive cumulative values.
            #
            # IMPORTANT: Group by fiscal-year start date, NOT by EDGAR's `fy` field.
            # EDGAR tags comparative/restated data from prior years with the current
            # filing's `fy`, so grouping by `fy` mixes fiscal years and produces
            # incorrect differences.
            cumulative = q_data[(q_data['duration_days'] > 120) & (q_data['duration_days'] < 400)].copy()
            if not cumulative.empty:
                cumulative = cumulative.sort_values(['end', 'filed']).drop_duplicates(subset=['end'], keep='last')
                cumulative = cumulative.sort_values('end')

                # Group by fiscal year START date (correctly separates fiscal years)
                for fy_start in cumulative['start'].dropna().unique():
                    fy_cums = cumulative[cumulative['start'] == fy_start].sort_values('end')
                    if fy_cums.empty:
                        continue

                    fy_end_approx = fy_cums['end'].max()

                    prev_cumulative_val = 0.0
                    # Check if we have a Q1 single-quarter value for this FY
                    for qe, (qv, _) in results.items():
                        if fy_start <= qe <= fy_end_approx:
                            days_from_start = (qe - fy_start).days
                            if days_from_start < 120:
                                prev_cumulative_val = qv
                                break

                    for _, cum_row in fy_cums.iterrows():
                        cum_end = cum_row['end']
                        if cum_end in results:
                            # Already have a single-quarter value, but still
                            # update prev_cumulative_val so subsequent diffs
                            # are correct
                            prev_cumulative_val = cum_row['val']
                            continue

                        q_val = cum_row['val'] - prev_cumulative_val
                        results[cum_end] = (q_val, cum_row['filed'])
                        prev_cumulative_val = cum_row['val']

    # --- Step 3: Derive Q4 from 10-K annual values ---
    if 'start' in raw.columns:
        annual_dur = raw.dropna(subset=['start', 'end']).copy()
        if not annual_dur.empty:
            annual_dur['duration_days'] = (annual_dur['end'] - annual_dur['start']).dt.days
            fy_rows = annual_dur[annual_dur['duration_days'] >= 300]
        else:
            fy_rows = pd.DataFrame()
    else:
        fy_rows = pd.DataFrame()

    if fy_rows.empty and 'fp' in raw.columns:
        annual = raw[raw['form'].isin(['10-K', '10-K/A'])]
        fy_rows = annual[annual['fp'] == 'FY']

    if not fy_rows.empty:
        fy_rows = fy_rows.sort_values('filed').drop_duplicates(subset=['end'], keep='last')
        for _, fy_row in fy_rows.iterrows():
            fy_end = fy_row['end']
            fy_val = fy_row['val']
            fy_filed = fy_row['filed']

            if fy_end in results:
                continue

            # Sum Q1+Q2+Q3 for this fiscal year
            q_sum = 0.0
            q_count = 0
            for qe, (qv, _) in results.items():
                days_before = (fy_end - qe).days
                if 0 < days_before <= 300:
                    q_sum += qv
                    q_count += 1

            if q_count >= 2:
                q4_val = fy_val - q_sum
                results[fy_end] = (q4_val, fy_filed)

    if not results:
        return pd.DataFrame(columns=['quarter_end', 'value', 'filed'])

    result_df = pd.DataFrame([
        {'quarter_end': k, 'value': v[0], 'filed': v[1]}
        for k, v in results.items()
    ])
    return result_df


def _snap_to_nearest(target_date, reference_dates, max_days=45):
    """Find the nearest reference date within max_days of target_date."""
    best = None
    best_dist = max_days + 1
    for ref in reference_dates:
        dist = abs((target_date - ref).days)
        if dist < best_dist:
            best_dist = dist
            best = ref
    return best


def _build_quarterly_dataframe(facts_data: dict) -> Optional[pd.DataFrame]:
    """
    Build a clean quarterly fundamentals DataFrame from EDGAR company facts.

    Uses duration fields (income statement) to establish the master quarter list,
    then snaps instant fields (balance sheet) to those dates. This prevents
    non-standard dates (e.g., cover page filing dates) from inflating the output.

    Returns DataFrame with all fundamental metrics per quarter.
    """
    duration_series = {}
    instant_series = {}

    # Extract income statement items (duration) FIRST to establish master quarters
    for field in DURATION_FIELDS:
        series = _extract_quarterly_duration(facts_data, field)
        if not series.empty:
            duration_series[field] = series

    # Extract balance sheet items (instant)
    for field in INSTANT_FIELDS:
        series = _extract_quarterly_instant(facts_data, field)
        if not series.empty:
            instant_series[field] = series

    if not duration_series and not instant_series:
        return None

    # Build master quarter list from duration fields (authoritative)
    master_quarters = set()
    filing_dates = {}  # quarter_end -> latest filing date

    for field, series in duration_series.items():
        for _, row in series.iterrows():
            qe = row['quarter_end']
            master_quarters.add(qe)
            if qe not in filing_dates or row['filed'] > filing_dates[qe]:
                filing_dates[qe] = row['filed']

    # If no duration fields, fall back to instant fields but filter to month-end dates
    if not master_quarters:
        for field, series in instant_series.items():
            for _, row in series.iterrows():
                qe = row['quarter_end']
                # Only accept dates near month-ends (day >= 25 or day <= 5 of next month)
                if qe.day >= 25 or qe.day <= 5:
                    master_quarters.add(qe)
                    if qe not in filing_dates or row['filed'] > filing_dates[qe]:
                        filing_dates[qe] = row['filed']

    if not master_quarters:
        return None

    sorted_quarters = sorted(master_quarters)

    # Also collect filing dates from instant fields for quarters they match
    for field, series in instant_series.items():
        for _, row in series.iterrows():
            snapped = _snap_to_nearest(row['quarter_end'], sorted_quarters, max_days=45)
            if snapped is not None and (snapped not in filing_dates or row['filed'] > filing_dates[snapped]):
                filing_dates[snapped] = row['filed']

    # Build rows
    rows = []
    for qe in sorted_quarters:
        row = {
            'quarter_end_date': qe,
            'report_date': filing_dates.get(qe, pd.NaT),
        }

        # Duration fields: exact match
        for field, series in duration_series.items():
            match = series[series['quarter_end'] == qe]
            if not match.empty:
                row[field] = match.iloc[0]['value']
            else:
                row[field] = np.nan

        # Instant fields: snap to nearest master quarter
        for field, series in instant_series.items():
            # Find the closest instant value within ±45 days of this quarter-end
            series_copy = series.copy()
            series_copy['_dist'] = (series_copy['quarter_end'] - qe).abs().dt.days
            close = series_copy[series_copy['_dist'] <= 45]
            if not close.empty:
                row[field] = close.sort_values('_dist').iloc[0]['value']
            else:
                row[field] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure schema consistency across tickers/runs.
    expected_fields = sorted(INSTANT_FIELDS | DURATION_FIELDS)
    for field in expected_fields:
        if field not in df.columns:
            df[field] = np.nan

    # Combine long-term + short-term debt into total_debt
    lt = df.get('long_term_debt', pd.Series(np.nan, index=df.index))
    st = df.get('short_term_debt', pd.Series(np.nan, index=df.index))
    # If both are NaN, total_debt is NaN; if one is NaN, treat as 0
    df['total_debt'] = lt.fillna(0) + st.fillna(0)
    df.loc[lt.isna() & st.isna(), 'total_debt'] = np.nan

    # Drop intermediate debt columns
    df = df.drop(columns=['long_term_debt', 'short_term_debt'], errors='ignore')

    # Calculate EBITDA = net_income + income_tax + interest_expense + depreciation
    ebitda_components = ['net_income', 'income_tax', 'interest_expense', 'depreciation']
    if all(c in df.columns for c in ebitda_components):
        df['ebitda'] = df[ebitda_components].sum(axis=1, min_count=1)
    else:
        df['ebitda'] = np.nan

    # Keep components for downstream proxy features.

    # Calculate ratios
    df['profit_margin'] = df.apply(
        lambda r: _safe_div(r.get('net_income'), r.get('revenue')), axis=1
    )
    df['operating_margin'] = df.apply(
        lambda r: _safe_div(r.get('operating_income'), r.get('revenue')), axis=1
    )
    df['gross_margin'] = df.apply(
        lambda r: _safe_div(r.get('gross_profit'), r.get('revenue')), axis=1
    )
    df['roe'] = df.apply(
        lambda r: _safe_div(r.get('net_income'), r.get('total_equity')), axis=1
    )
    df['roa'] = df.apply(
        lambda r: _safe_div(r.get('net_income'), r.get('total_assets')), axis=1
    )
    df['debt_to_equity'] = df.apply(
        lambda r: _safe_div(r.get('total_debt'), r.get('total_equity')), axis=1
    )
    df['current_ratio'] = df.apply(
        lambda r: _safe_div(r.get('current_assets'), r.get('current_liabilities')), axis=1
    )

    inv = df.get('inventory', pd.Series(np.nan, index=df.index))
    ca = df.get('current_assets', pd.Series(np.nan, index=df.index))
    cl = df.get('current_liabilities', pd.Series(np.nan, index=df.index))
    df['quick_ratio'] = (ca - inv.fillna(0)) / cl.replace(0, np.nan)

    # Keep inventory for downstream GKX proxy features.

    # Additional helper aliases used by feature layer
    df['book_value'] = df.get('total_equity')
    df['asset_turnover'] = df.apply(
        lambda r: _safe_div(r.get('revenue'), r.get('total_assets')), axis=1
    )

    return df


# ---------------------------------------------------------------------------
# Public API (same signatures as before)
# ---------------------------------------------------------------------------

def fetch_quarterly_fundamentals(ticker: str, reporting_lag_days: int = 45) -> Optional[pd.DataFrame]:
    """
    Fetch quarterly fundamentals from SEC EDGAR with actual filing dates.

    Args:
        ticker: Stock ticker symbol
        reporting_lag_days: IGNORED - kept for API compatibility.
                           Actual SEC filing dates are used instead.

    Returns:
        DataFrame with columns: ticker, quarter_end_date, report_date, and all fundamental metrics
    """
    try:
        # Map ticker to CIK
        cik_map = _load_cik_mapping()
        cik = cik_map.get(ticker.upper())
        if cik is None:
            print(f"  No CIK found for {ticker}")
            return None

        # Fetch company facts from EDGAR
        facts_data = _fetch_company_facts(cik)
        if facts_data is None:
            print(f"  No EDGAR data for {ticker} (CIK: {cik})")
            return None

        # Build quarterly DataFrame
        df = _build_quarterly_dataframe(facts_data)
        if df is None or df.empty:
            print(f"  No quarterly data extracted for {ticker}")
            return None

        # Add ticker and sort
        df['ticker'] = ticker
        df = df.sort_values('quarter_end_date').reset_index(drop=True)

        # Calculate growth metrics defensively to avoid dropping tickers with
        # partial disclosures in some fields.
        revenue = df['revenue'] if 'revenue' in df.columns else pd.Series(np.nan, index=df.index)
        net_income = df['net_income'] if 'net_income' in df.columns else pd.Series(np.nan, index=df.index)
        df['revenue_growth_qoq'] = revenue.pct_change()
        df['earnings_growth_qoq'] = net_income.pct_change()
        df['revenue_growth_yoy'] = revenue.pct_change(periods=4)
        df['earnings_growth_yoy'] = net_income.pct_change(periods=4)

        # Format dates as strings for output consistency
        df['quarter_end_date'] = df['quarter_end_date'].dt.strftime('%Y-%m-%d')
        df['report_date'] = df['report_date'].dt.strftime('%Y-%m-%d')

        return df

    except Exception as e:
        print(f"  Error fetching quarterly fundamentals for {ticker}: {e}")
        return None


def fetch_current_valuation_metrics(ticker: str) -> Dict:
    """
    Fetch current valuation metrics that change daily with stock price.
    These need to be calculated separately since P/E, P/B etc. depend on current price.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with current valuation metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            'ticker': ticker,
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'beta': info.get('beta'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'float_shares': info.get('floatShares'),
            'held_percent_insiders': info.get('heldPercentInsiders'),
            'held_percent_institutions': info.get('heldPercentInstitutions'),
            'dividend_yield': info.get('dividendYield'),
            'book_value': info.get('bookValue'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'trailing_pe': info.get('trailingPE'),
        }

    except Exception as e:
        print(f"  Error fetching valuation for {ticker}: {e}")
        return None


def fetch_all_quarterly_fundamentals(tickers: List[str],
                                     save_dir: str = "data/raw/fundamentals_quarterly/",
                                     reporting_lag_days: int = 45) -> pd.DataFrame:
    """
    Fetch quarterly fundamentals from SEC EDGAR for a list of tickers.

    Args:
        tickers: List of stock ticker symbols
        save_dir: Directory to save data
        reporting_lag_days: IGNORED - kept for API compatibility

    Returns:
        DataFrame with all quarterly fundamentals
    """
    os.makedirs(save_dir, exist_ok=True)

    # Pre-load CIK mapping
    _load_cik_mapping()

    all_data = []
    failed_tickers = []

    print(f"Fetching quarterly fundamentals from SEC EDGAR for {len(tickers)} tickers...")
    print(f"Using actual SEC filing dates (no estimated lag)")

    for i, ticker in enumerate(tickers):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(tickers)}")

        df = fetch_quarterly_fundamentals(ticker)

        if df is not None and not df.empty:
            all_data.append(df)
        else:
            failed_tickers.append(ticker)

        # Rate limiting is handled inside _fetch_company_facts
        # Small additional delay for safety margin
        time.sleep(0.02)

    if not all_data:
        print("No data fetched!")
        return pd.DataFrame()

    # Combine all data
    fundamentals_df = pd.concat(all_data, ignore_index=True)

    # Save
    output_path = os.path.join(save_dir, 'quarterly_fundamentals.parquet')
    fundamentals_df.to_parquet(output_path, index=False)

    # Save failed tickers
    if failed_tickers:
        failed_path = os.path.join(save_dir, 'failed_tickers.json')
        with open(failed_path, 'w') as f:
            json.dump(failed_tickers, f, indent=2)
        print(f"\nWarning: Failed to fetch {len(failed_tickers)} tickers")

    print(f"\n=== Summary ===")
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

    For each date in price_df, we use the most recent fundamentals that were
    REPORTED (available) before that date. This prevents look-ahead bias.

    Args:
        price_df: DataFrame with columns ['ticker', 'date', ...]
        fundamentals_df: DataFrame with quarterly fundamentals including 'report_date'
        valuation_df: Optional DataFrame with sector/industry info (merged on ticker)

    Returns:
        Merged DataFrame with fundamentals correctly aligned
    """
    print("Performing point-in-time merge of fundamentals...")

    # Ensure date columns are datetime
    price_df = price_df.copy()
    fundamentals_df = fundamentals_df.copy()

    price_df['date'] = pd.to_datetime(price_df['date']).dt.as_unit('ns')
    fundamentals_df['report_date'] = pd.to_datetime(fundamentals_df['report_date']).dt.as_unit('ns')
    fundamentals_df['quarter_end_date'] = pd.to_datetime(fundamentals_df['quarter_end_date']).dt.as_unit('ns')

    # merge_asof requires left_on and right_on to be globally sorted
    price_df = price_df.sort_values('date').reset_index(drop=True)
    fundamentals_df = fundamentals_df.sort_values('report_date').reset_index(drop=True)

    # Perform as-of merge (merge_asof)
    # For each ticker/date in price_df, get the most recent fundamentals where report_date <= date
    merged = pd.merge_asof(
        price_df,
        fundamentals_df,
        left_on='date',
        right_on='report_date',
        by='ticker',
        direction='backward',
        suffixes=('', '_fundamental')
    )

    # Restore ticker+date sort order
    merged = merged.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Merge valuation metrics (sector, industry) if provided
    if valuation_df is not None:
        val_cols = [c for c in valuation_df.columns if c != 'ticker']
        merged = merged.merge(valuation_df[['ticker'] + val_cols], on='ticker', how='left')

    print(f"Merged {len(merged)} rows")
    print(f"Fundamentals coverage: {merged['quarter_end_date'].notna().sum() / len(merged) * 100:.1f}%")

    # Report any missing fundamentals
    missing_pct = merged['quarter_end_date'].isna().sum() / len(merged) * 100
    if missing_pct > 0:
        print(f"Warning: {missing_pct:.1f}% of rows have no fundamentals (early dates or delisted stocks)")

    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch quarterly fundamentals from SEC EDGAR")
    parser.add_argument("--tickers", type=str, required=True,
                       help="Comma-separated list of tickers OR path to JSON file")
    parser.add_argument("--data_tag", type=str, default=None,
                       help="Data tag for output directory (default: auto-generated timestamp)")

    args = parser.parse_args()

    # Generate data tag
    data_tag = args.data_tag or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"data/raw/{data_tag}/fundamentals_quarterly/"

    # Parse tickers
    if args.tickers.endswith('.json'):
        with open(args.tickers, 'r') as f:
            data = json.load(f)
            tickers = data if isinstance(data, list) else data.get('tickers', [])
    else:
        tickers = args.tickers.split(',')

    print(f"Fetching quarterly fundamentals for {len(tickers)} tickers...")
    print(f"Data tag: {data_tag}")
    print(f"Output dir: {save_dir}")

    # Fetch quarterly fundamentals
    fundamentals_df = fetch_all_quarterly_fundamentals(
        tickers,
        save_dir=save_dir,
    )

    print("\n=== Sample Data ===")
    print(fundamentals_df.head(10))

    print("\n=== Data Completeness ===")
    completeness = (1 - fundamentals_df.isnull().sum() / len(fundamentals_df)) * 100
    print(completeness.sort_values(ascending=False))
