# src/data_fetch/get_universe.py
"""
Build a large US tradable universe and rank securities by liquidity.

Preserves original API but:
 - expands symbol sources (nasdaq + sec + etfdb fallback)
 - robust ETF detection (nasdaq flag + yfinance + heuristics)
 - batched yfinance.download for speed (batch_size, parallel workers)
 - thread-based parallelism + retry/backoff
"""
import os
import json
import time
import random
import requests
from typing import List, Dict, Optional, Tuple, Iterable, Set
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from pathlib import Path

try:
    from yfinance.exceptions import YFRateLimitError
except Exception:
    class YFRateLimitError(Exception):
        pass

# ---------------------------
# Configurable defaults
# ---------------------------
DEFAULT_BATCH_SIZE = 10  # number of tickers per yf.download call
DEFAULT_WORKERS = min(1, cpu_count() - 1) # thread workers for batches
DEFAULT_LOOKBACK_DAYS = 60
DEFAULT_MIN_OBS = 20
CACHE_DIR = Path("data/universe_cache")
ETF_KEYWORDS = [
    "ETF", "Fund", "Trust", "Shares", "SPDR", "iShares", "Vanguard",
    "ProShares", "Direxion", "ARK", "Invesco", "DIREXION", "ISHARES"
]
BAD_SUFFIXES = {"W", "WS", "U", "R", "RT", "V", "P", "PR", "A", "B", "C", "D"}  # expand as needed
MAX_YF_RETRIES = 3
YF_BACKOFF_BASE = 1.2

# Ensure cache dir
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")

def _is_excluded_symbol(ticker: str) -> bool:
    """Exclude non-common-share suffixes and obviously-bad entries."""
    if not isinstance(ticker, str):
        return True
    t = ticker.strip().upper()
    if not t:
        return True
    if t.startswith("TEST") or t.endswith(".TEST"):
        return True
    if len(t) > 7:  # defensively allow up to 7 (e.g., BRK-B)
        return True
    # Only alphanumeric + hyphen
    if not t.replace("-", "").isalnum():
        return True
    # Suffix-based filtering (warrants, units, preferred)
    if "-" in t:
        suffix = t.split("-")[-1]
        if suffix in BAD_SUFFIXES:
            return True
    return False

# ---------------------------
# Fetch listings (expanded sources)
# ---------------------------
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
ETFDB_URL = "https://etfdb.com/etf/"  # used as heuristic scraping fallback

def get_all_us_symbols_with_metadata() -> pd.DataFrame:
    """
    Fetch US-listed symbols from multiple authoritative sources and return DataFrame:
      ticker, is_etf (from source flags, False default)
    """
    listings_frames = []

    # 1) NASDAQ official files (nasdaqlisted + otherlisted)
    try:
        nas = pd.read_csv(NASDAQ_URL, sep="|", dtype=str)
        nas = nas[nas["Symbol"].notna()]
        nas_df = pd.DataFrame({
            "ticker": nas["Symbol"].map(_normalize_ticker),
            "is_etf": nas.get("ETF", "N").astype(str).str.upper().eq("Y")
        })
        listings_frames.append(nas_df)
    except Exception:
        # continue gracefully
        pass

    try:
        other = pd.read_csv(OTHER_URL, sep="|", dtype=str)
        other = other[other["ACT Symbol"].notna()]
        other_df = pd.DataFrame({
            "ticker": other["ACT Symbol"].map(_normalize_ticker),
            "is_etf": other.get("ETF", "N").astype(str).str.upper().eq("Y")
        })
        listings_frames.append(other_df)
    except Exception:
        pass

    # 2) SEC mapping (covers tickers -> CIK); use as another source
    try:
        resp = requests.get(SEC_TICKERS_URL, timeout=10)
        if resp.ok:
            data = resp.json()
            rows = []
            for v in data.values():
                t = v.get("ticker", "")
                if not t:
                    continue
                rows.append({"ticker": _normalize_ticker(t), "is_etf": False})
            if rows:
                listings_frames.append(pd.DataFrame(rows))
    except Exception:
        pass

    # 3) ETFDB (best-effort): try to get ETF tickers to augment ETF labeling
    etf_list = []
    try:
        # pandas.read_html can often pull a large ETF listing; if it fails, ignore
        etf_tables = pd.read_html(ETFDB_URL)
        # find any table with 'Ticker' or 'Symbol' column
        for tbl in etf_tables:
            for col in ("Ticker", "Symbol"):
                if col in tbl.columns:
                    etf_list += tbl[col].astype(str).str.strip().tolist()
        etf_list = [_normalize_ticker(t) for t in etf_list]
        if etf_list:
            etf_df = pd.DataFrame({"ticker": etf_list, "is_etf": True})
            listings_frames.append(etf_df)
    except Exception:
        # not critical
        pass

    if not listings_frames:
        print("Warning: no listing sources succeeded; returning empty DataFrame")
        return pd.DataFrame(columns=["ticker", "is_etf"])

    # combine, deduplicate, prefer is_etf True if any source says so
    all_listings = pd.concat(listings_frames, ignore_index=True)
    all_listings = all_listings.dropna(subset=["ticker"])
    all_listings = all_listings[~all_listings["ticker"].map(_is_excluded_symbol)]
    all_listings = all_listings.assign(
        ticker=lambda d: d["ticker"].map(lambda x: _normalize_ticker(x))
    )
    listings = (
        all_listings.groupby("ticker", as_index=False)["is_etf"]
        .max()
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    print(f"Fetched {len(listings)} candidate US symbols from combined sources")
    return listings

def get_all_us_symbols() -> List[str]:
    return get_all_us_symbols_with_metadata()["ticker"].tolist()

# ---------------------------
# ETF detection (robust)
# ---------------------------
_ETF_DETECTION_CACHE = {}

def _detect_etf_yf(ticker: str) -> bool:
    """
    Detect ETF with yfinance: check fast_info.quoteType or info['quoteType'],
    then fallback to name/keyword heuristics.
    """
    t = _normalize_ticker(ticker)
    if t in _ETF_DETECTION_CACHE:
        return _ETF_DETECTION_CACHE[t]

    is_etf = False
    try:
        tk = yf.Ticker(t)
        # try fast_info first (lightweight)
        fast = getattr(tk, "fast_info", None)
        if isinstance(fast, dict) and fast.get("quoteType", "").upper() == "ETF":
            is_etf = True
        else:
            info = {}
            try:
                info = tk.info or {}
            except Exception:
                info = {}
            qt = str(info.get("quoteType", "")).upper()
            if qt == "ETF" or qt == "MUTUALFUND":
                is_etf = True
            else:
                # check longName/shortName for ETF keywords
                name = (str(info.get("longName", "") or "") + " " + str(info.get("shortName", "") or "")).upper()
                if any(k.upper() in name for k in ETF_KEYWORDS):
                    is_etf = True
    except Exception:
        # conservative default: not ETF if detection fails
        is_etf = False

    _ETF_DETECTION_CACHE[t] = is_etf
    return is_etf

def classify_etfs_parallel(df: pd.DataFrame, n_workers: int = 8) -> pd.DataFrame:
    """
    Parallel ETF classification for rows missing is_etf==True from source.
    Mutates and returns a new DataFrame with 'is_etf' column updated.
    """
    tickers = df["ticker"].tolist()
    results = {}
    # Use threads for I/O bound yfinance calls
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_detect_etf_yf, t): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ETF detect"):
            t = futures[fut]
            try:
                results[t] = fut.result()
            except Exception:
                results[t] = False
    df = df.copy()
    df["is_etf_yf"] = df["ticker"].map(lambda t: results.get(t, False))
    # combine: treat as ETF if either source or yfinance says so
    df["is_etf"] = df["is_etf"] | df["is_etf_yf"]
    return df

# ---------------------------
# Batched liquidity computation
# ---------------------------
def _chunk_iterable(iterable: Iterable, size: int) -> Iterable[List]:
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

def _compute_batch_liquidity(batch: List[str], lookback_days: int, min_obs: int) -> List[Dict]:
    """
    Download a batch of tickers via yf.download and compute avg price/volume.
    Returns list of dict rows.
    """
    rows = []
    if not batch:
        return rows
    # retry wrapper for yf.download per batch
    for attempt in range(MAX_YF_RETRIES):
        try:
            # group_by='ticker' will create multi-columns
            df = yf.download(
                tickers=batch,
                period=f"{lookback_days}d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception as e:
            if attempt < MAX_YF_RETRIES - 1:
                time.sleep((YF_BACKOFF_BASE ** (attempt + 1)) + random.uniform(0.2, 0.6))
                continue
            else:
                # irrecoverable; return empty
                return rows

    # if single-ticker, yfinance returns simple DataFrame (no outermost ticker level)
    # normalize to dict mapping ticker -> DataFrame
    if df is None or df.empty:
        # nothing downloaded
        return rows

    # If the DataFrame has columns as single-level (single ticker), yfinance sometimes
    # returns e.g. columns: Close, Volume - which means one ticker requested or mapping different
    # We'll attempt to find ticker subframes. Use columns MultiIndex detection:
    if isinstance(df.columns, pd.MultiIndex):
        tickers_in_df = df.columns.levels[0].tolist()
        for t in batch:
            if t not in tickers_in_df:
                continue
            try:
                sdf = df[t]
                if sdf.empty or len(sdf) < min_obs:
                    continue
                avg_price = float(sdf["Close"].mean())
                avg_volume = float(sdf["Volume"].mean())
                rows.append({
                    "ticker": t,
                    "avg_price": avg_price,
                    "avg_volume": avg_volume,
                    "avg_dollar_volume": avg_price * avg_volume
                })
            except Exception:
                continue
    else:
        # single-level columns: assume only one ticker in the batch had data; try to map by heuristics
        # iterate through batch and attempt individual downloads fallback
        for t in batch:
            try:
                sdf = df.copy()
                if sdf.empty or len(sdf) < min_obs:
                    continue
                # assume df corresponds to the first ticker with data (best-effort)
                avg_price = float(sdf["Close"].mean())
                avg_volume = float(sdf["Volume"].mean())
                rows.append({
                    "ticker": t,
                    "avg_price": avg_price,
                    "avg_volume": avg_volume,
                    "avg_dollar_volume": avg_price * avg_volume
                })
            except Exception:
                continue

    return rows

def compute_liquidity_parallel(
    symbols: List[str],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_obs: int = DEFAULT_MIN_OBS,
    n_workers: int = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Compute liquidity using batched yfinance.download calls in parallel threads.

    Returns DataFrame with ticker, avg_price, avg_volume, avg_dollar_volume
    """
    symbols = [s for s in symbols if not _is_excluded_symbol(s)]
    if n_workers is None:
        n_workers = DEFAULT_WORKERS

    print(f"Computing liquidity for {len(symbols)} symbols using {n_workers} workers, batch_size={batch_size}...")

    all_batches = list(_chunk_iterable(symbols, batch_size))
    results = []

    # Warm up YF API to avoid first-call throttling
    try:
        yf.download("SPY", period="5d", progress=False)
    except Exception:
        pass

    # Use threadpool to parallelize batches (I/O bound)
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        future_to_batch = {exe.submit(_compute_batch_liquidity, batch, lookback_days, min_obs): batch for batch in all_batches}
        for fut in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Processing batches"):
            try:
                batch_rows = fut.result()
                if batch_rows:
                    results.extend(batch_rows)
            except Exception:
                # skip failing batch
                continue

            time.sleep(random.uniform(1.5, 3.0))


    df = pd.DataFrame(results)
    print(f"Successfully computed liquidity for {len(df)} symbols")
    return df

# ---------------------------
# Utility: compute liquidity for explicit tickers
# ---------------------------
def _compute_liquidity_for_tickers(
    tickers: List[str],
    lookback_days: int,
    min_obs: int,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "avg_price", "avg_volume", "avg_dollar_volume"])
    # reuse batch function but single batch
    rows = _compute_batch_liquidity(tickers, lookback_days, min_obs)
    return pd.DataFrame(rows)

# ---------------------------
# Index helpers preserved (SP500, NASDAQ100 approximations)
# ---------------------------
def get_sp500_tickers_from_etf() -> List[str]:
    """Attempt to fetch S&P 500 holdings from SPY via yfinance (best-effort)."""
    try:
        spy = yf.Ticker("SPY")
        # try multiple access patterns for holdings, depending on yfinance version
        holdings = None
        try:
            holdings = spy.get_holdings()
        except Exception:
            pass
        if holdings is None:
            try:
                holdings = spy.holdings
            except Exception:
                holdings = None
        if holdings is not None:
            # holdings may be DataFrame or dict-like
            if isinstance(holdings, pd.DataFrame):
                tickers = holdings.index.astype(str).tolist()
            elif isinstance(holdings, dict):
                tickers = list(holdings.keys())
            else:
                tickers = []
            tickers = [_normalize_ticker(t) for t in tickers]
            return tickers
    except Exception:
        pass
    return []

def get_sp500_tickers() -> List[str]:
    """Wikipedia fallback for S&P 500 constituents (best-effort)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        for table in tables:
            if "Symbol" in table.columns and "Security" in table.columns:
                tickers = table["Symbol"].astype(str).str.strip().tolist()
                tickers = [_normalize_ticker(t.replace(".", "-")) for t in tickers if t and t != "nan"]
                if len(tickers) >= 400:
                    return tickers
    except Exception:
        pass
    return get_sp500_tickers_from_etf()

def get_nasdaq100_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = pd.read_html(url)
        for table in tables:
            if "Ticker" in table.columns and "Company" in table.columns:
                tickers = table["Ticker"].astype(str).str.strip().tolist()
                tickers = [_normalize_ticker(t.replace(".", "-")) for t in tickers if t and t != "nan"]
                if len(tickers) >= 80:
                    return tickers
    except Exception:
        pass
    return []

def get_russell1000_approximation() -> List[str]:
    sp500 = get_sp500_tickers()
    nas100 = get_nasdaq100_tickers()
    all_tickers = list(set(sp500 + nas100))
    return all_tickers

# ---------------------------
# Top-N universe composition (preserve API)
# ---------------------------
def get_top_n_equities_by_liquidity(
    n: int = 1000,
    save_path: str = "data/universe/",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    n_workers: int = None,
    use_nasdaq_source: bool = True,
    exclude_etfs: bool = True,
    force_include_tickers: Optional[List[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    classify_etfs: bool = True,
    etf_detection_workers: int = 8,
) -> pd.DataFrame:
    """
    Get the top N US equities by liquidity (avg dollar volume).

    Preserves original API but with improved universes and ETF detection.
    """
    os.makedirs(save_path, exist_ok=True)

    print("\n=== Step 1: Fetching all US symbols ===")
    listings_df = pd.DataFrame(columns=["ticker", "is_etf"])
    if use_nasdaq_source:
        listings_df = get_all_us_symbols_with_metadata()
        if listings_df.empty:
            print("NASDAQ/SEC sources failed; falling back to index approximations")
            all_symbols = get_russell1000_approximation()
        else:
            all_symbols = listings_df["ticker"].tolist()
    else:
        all_symbols = get_russell1000_approximation()

    # if user requested exclude_etfs early, filter by listed flag for speed (we'll re-evaluate later)
    if exclude_etfs and not listings_df.empty:
        all_symbols = listings_df[~listings_df["is_etf"]]["ticker"].tolist()

    if not all_symbols:
        raise ValueError("Failed to fetch any tickers. Check network and sources.")

    print(f"Found {len(all_symbols)} candidate symbols (pre-filter)")

    # Step 2: compute liquidity in parallel (batched)
    print("\n=== Step 2: Computing liquidity metrics ===")
    liquidity_df = compute_liquidity_parallel(
        all_symbols,
        lookback_days=lookback_days,
        min_obs=DEFAULT_MIN_OBS,
        n_workers=n_workers,
        batch_size=batch_size,
    )

    if liquidity_df.empty:
        raise ValueError("No tickers successfully fetched liquidity data! Check yfinance/network.")

    # Step 2b: merge in is_etf metadata and finish ETF classification if requested
    if not listings_df.empty:
        merged = liquidity_df.merge(listings_df, on="ticker", how="left")
        merged["is_etf"] = merged["is_etf"].fillna(False)
    else:
        merged = liquidity_df.copy()
        merged["is_etf"] = False

    # Additional ETF detection for tickers not flagged (best-effort)
    if classify_etfs:
        print("Running additional ETF detection (yfinance + heuristics)...")
        merged = classify_etfs_parallel(merged, n_workers=etf_detection_workers)

    # Step 3: filter out ETFs if requested
    if exclude_etfs:
        merged = merged[~merged["is_etf"]].copy()
        print(f"After excluding ETFs: {len(merged)} symbols remain")

    # Step 4: sort by avg_dollar_volume and pick top N
    ranked = merged.sort_values("avg_dollar_volume", ascending=False).reset_index(drop=True)
    top_df = ranked.head(n).copy()
    top_df["forced_include"] = False

    # Step 4b: ensure forced-included tickers appear (compute liquidity for missing forced ones)
    if force_include_tickers:
        force_set = {_normalize_ticker(t) for t in force_include_tickers if str(t).strip()}
        if force_set:
            missing = [t for t in force_set if t not in set(top_df["ticker"])]
            if missing:
                print(f"Fetching liquidity for forced-included tickers: {missing}")
                extra = _compute_liquidity_for_tickers(missing, lookback_days=lookback_days, min_obs=DEFAULT_MIN_OBS)
                if not extra.empty:
                    # merge ETF flags for these
                    if not listings_df.empty:
                        extra = extra.merge(listings_df, on="ticker", how="left")
                        extra["is_etf"] = extra["is_etf"].fillna(False)
                    extra["forced_include"] = True
                    top_df = pd.concat([top_df, extra], ignore_index=True)
            # mark forced_included if present
            top_df.loc[top_df["ticker"].isin(force_set), "forced_include"] = True

    # final ranking and rank column
    top_df = top_df.sort_values("avg_dollar_volume", ascending=False).reset_index(drop=True)
    top_df["rank"] = range(1, len(top_df) + 1)

    # Save results (CSV + JSON)
    if exclude_etfs:
        csv_path = os.path.join(save_path, f"top_{n}_equities_by_liquidity_non_etf.csv")
        json_path = os.path.join(save_path, f"top_{n}_tickers_non_etf.json")
    else:
        csv_path = os.path.join(save_path, f"top_{n}_equities_by_liquidity.csv")
        json_path = os.path.join(save_path, f"top_{n}_tickers.json")

    top_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(top_df["ticker"].tolist(), f, indent=2)

    # Print summary
    print("\n=== Summary ===")
    print(f"Top {len(top_df)} equities by liquidity identified")
    try:
        print(f"Avg dollar volume range: ${top_df['avg_dollar_volume'].min():,.0f} - ${top_df['avg_dollar_volume'].max():,.0f}")
        print(f"Avg volume range: {top_df['avg_volume'].min():,.0f} - {top_df['avg_volume'].max():,.0f}")
        print(f"Avg price range: ${top_df['avg_price'].min():.2f} - ${top_df['avg_price'].max():,.2f}")
    except Exception:
        pass
    print(f"Saved to:\n  - {csv_path}\n  - {json_path}")

    # Print top 10 nicely
    print("\nTop 10 by liquidity:")
    display_cols = ["rank", "ticker", "avg_dollar_volume", "avg_volume", "avg_price", "forced_include"]
    display_df = top_df[display_cols].head(10).copy()
    display_df["avg_dollar_volume"] = display_df["avg_dollar_volume"].apply(lambda x: f"${x:,.0f}")
    display_df["avg_volume"] = display_df["avg_volume"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "NA")
    display_df["avg_price"] = display_df["avg_price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "NA")
    print(display_df.to_string(index=False))

    return top_df

# ---------------------------
# load_universe (preserve API)
# ---------------------------
def load_universe(path: str = "data/universe/top_1000_tickers.json") -> List[str]:
    import glob
    from pathlib import Path
    if Path(path).exists() is False:
        replace_str = path.split("/")[-1].replace("json", "")
        glob_path = path.replace(replace_str, "*")
        tickers_json_paths = glob.glob(glob_path)
        if len(tickers_json_paths) > 0:
            path = tickers_json_paths[0]
        else:
            raise ValueError("No top tickers json file found")
    with open(path, "r") as f:
        tickers = json.load(f)
    print(f"Loaded {len(tickers)} tickers from {path}")
    return tickers

# ---------------------------
# CLI (preserve behavior)
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build liquid US equity universe")
    parser.add_argument("--n", type=int, default=1000, help="Number of equities to rank")
    parser.add_argument("--lookback_days", type=int, default=60, help="Liquidity lookback window")
    parser.add_argument("--exclude_etfs", action="store_true", help="Exclude ETFs from ranked universe")
    parser.add_argument("--force_include_tickers", type=str, default="SPY,QQQ,IWM", help="Comma-separated benchmark tickers to force-include")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of tickers per batch yf.download")
    parser.add_argument("--workers", type=int, default=None, help="Thread workers for batch downloads")
    parser.add_argument("--classify_etfs", action="store_true", help="Run additional ETF detection via yfinance")
    parser.add_argument("--etf_workers", type=int, default=4, help="Workers for ETF classification")
    args = parser.parse_args()

    force_include = [
        _normalize_ticker(t)
        for t in (args.force_include_tickers or "").split(",")
        if str(t).strip()
    ]

    get_top_n_equities_by_liquidity(
        n=args.n,
        lookback_days=args.lookback_days,
        exclude_etfs=args.exclude_etfs,
        n_workers=args.workers,
        force_include_tickers=force_include,
        batch_size=args.batch_size,
        classify_etfs=args.classify_etfs,
        etf_detection_workers=args.etf_workers,
    )
