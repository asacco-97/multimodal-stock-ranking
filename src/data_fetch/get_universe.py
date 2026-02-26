# src/data_fetch/get_universe.py
"""
Module to identify and fetch the top N US equities by liquidity (trading volume).
Uses NASDAQ official data and parallel processing for fast, accurate results.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Optional
import requests
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def get_all_us_symbols_with_metadata() -> pd.DataFrame:
    """
    Fetch all US-listed symbols from NASDAQ official symbol files with ETF flags.

    Returns:
        DataFrame with columns: ticker, is_etf
    """
    try:
        NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

        print("Fetching all US symbols from NASDAQ official data...")

        nasdaq = pd.read_csv(NASDAQ_URL, sep="|")
        nasdaq = nasdaq[nasdaq["Symbol"].notna()].copy()
        nasdaq = nasdaq[nasdaq["Symbol"].astype(str).str.strip() != ""].copy()
        nasdaq_df = pd.DataFrame({
            "ticker": nasdaq["Symbol"].map(_normalize_ticker),
            "is_etf": nasdaq.get("ETF", "N").astype(str).str.upper().eq("Y"),
        })

        other = pd.read_csv(OTHER_URL, sep="|")
        other = other[other["ACT Symbol"].notna()].copy()
        other = other[other["ACT Symbol"].astype(str).str.strip() != ""].copy()
        other_df = pd.DataFrame({
            "ticker": other["ACT Symbol"].map(_normalize_ticker),
            "is_etf": other.get("ETF", "N").astype(str).str.upper().eq("Y"),
        })

        listings = pd.concat([nasdaq_df, other_df], ignore_index=True)

        # Filter out test symbols and malformed symbols.
        listings = listings[
            (~listings["ticker"].str.endswith(".TEST", na=False))
            & (~listings["ticker"].str.startswith("TEST", na=False))
            & (listings["ticker"].str.len() <= 6)
            & (listings["ticker"].str.replace("-", "", regex=False).str.isalnum())
        ].copy()

        # Keep one row per ticker; if any source marks ETF, treat as ETF.
        listings = (
            listings.groupby("ticker", as_index=False)["is_etf"]
            .max()
            .sort_values("ticker")
            .reset_index(drop=True)
        )

        print(f"Fetched {len(listings)} US symbols from official NASDAQ data")
        return listings

    except Exception as e:
        print(f"Error fetching symbols from NASDAQ: {e}")
        return pd.DataFrame(columns=["ticker", "is_etf"])


def get_all_us_symbols() -> List[str]:
    """
    Fetch all US-listed stock symbols from NASDAQ official data files.
    This includes NASDAQ, NYSE, AMEX, and other exchanges.

    Returns:
        List of all US stock symbols
    """
    listings = get_all_us_symbols_with_metadata()
    return listings["ticker"].tolist()

def chunk_list(lst: List, n_chunks: int) -> List[List]:
    """Split a list into n roughly equal chunks."""
    return [chunk.tolist() for chunk in np.array_split(lst, n_chunks)]

def process_batch(args) -> List[Dict]:
    """
    Process a batch of symbols to compute liquidity metrics.

    Args:
        args: Tuple of (symbol_batch, lookback_days, min_obs)

    Returns:
        List of dicts with liquidity data
    """
    symbol_batch, lookback_days, min_obs = args
    results = []

    for sym in symbol_batch:
        try:
            df = yf.download(
                sym,
                period=f"{lookback_days}d",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if df.empty or len(df) < min_obs:
                continue

            # Ensure we extract scalar values, not Series
            # Handle both single-ticker and multi-ticker downloads
            if isinstance(df["Close"], pd.Series):
                avg_price = float(df["Close"].mean())
                avg_volume = float(df["Volume"].mean())
            else:
                # In case of MultiIndex columns
                avg_price = float(df["Close"].iloc[:, 0].mean() if hasattr(df["Close"], 'iloc') else df["Close"].mean())
                avg_volume = float(df["Volume"].iloc[:, 0].mean() if hasattr(df["Volume"], 'iloc') else df["Volume"].mean())

            dollar_volume = avg_price * avg_volume

            results.append({
                "ticker": sym,
                "avg_price": avg_price,
                "avg_volume": avg_volume,
                "avg_dollar_volume": dollar_volume
            })

        except Exception as e:
            # Silently skip failed symbols
            continue

    return results


def _compute_liquidity_for_tickers(
    tickers: List[str],
    lookback_days: int,
    min_obs: int,
) -> pd.DataFrame:
    """
    Compute liquidity for an explicit ticker subset (used for forced includes).
    """
    if not tickers:
        return pd.DataFrame(columns=["ticker", "avg_price", "avg_volume", "avg_dollar_volume"])
    rows = process_batch((tickers, lookback_days, min_obs))
    return pd.DataFrame(rows)

def compute_liquidity_parallel(
    symbols: List[str],
    lookback_days: int = 60,
    min_obs: int = 20,
    n_workers: int = None
) -> pd.DataFrame:
    """
    Compute liquidity metrics for symbols in parallel.

    Args:
        symbols: List of stock symbols
        lookback_days: Number of days to look back for volume calculation
        min_obs: Minimum number of observations required
        n_workers: Number of parallel workers (None = auto-detect)

    Returns:
        DataFrame with liquidity metrics
    """
    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)

    print(f"Computing liquidity for {len(symbols)} symbols using {n_workers} workers...")

    # Split symbols into batches
    symbol_batches = chunk_list(symbols, n_workers * 4)  # More batches than workers for better load balancing

    # Prepare arguments for each batch
    batch_args = [(batch, lookback_days, min_obs) for batch in symbol_batches]

    all_results = []

    with Pool(processes=n_workers) as pool:
        # Use imap_unordered for progress tracking
        for batch_result in tqdm(
            pool.imap_unordered(process_batch, batch_args),
            total=len(batch_args),
            desc="Processing batches"
        ):
            all_results.extend(batch_result)

    df = pd.DataFrame(all_results)
    print(f"Successfully computed liquidity for {len(df)} symbols")

    return df

def get_sp500_tickers_from_etf() -> List[str]:
    """
    Get S&P 500 tickers by fetching holdings from SPY ETF.
    More reliable than Wikipedia scraping.
    """
    try:
        print("Fetching S&P 500 tickers from SPY ETF...")
        spy = yf.Ticker("SPY")

        # Try to get holdings - this might not work with all yfinance versions
        # If this fails, we'll fall back to a smaller set
        try:
            # Some versions of yfinance have this
            holdings = spy.get_holdings()
            if holdings is not None and not holdings.empty:
                tickers = holdings.index.tolist()
                print(f"Fetched {len(tickers)} tickers from SPY holdings")
                return tickers
        except:
            pass

        # Fallback: Get major large cap stocks using yfinance Screener
        print("SPY holdings not available, fetching major US stocks...")

        # Get a broad list of US stocks by fetching from major index ETFs
        etfs_to_check = ['SPY', 'QQQ', 'IWM', 'DIA', 'IWF', 'IWD']
        all_tickers = set()

        for etf in etfs_to_check[:3]:  # Check first 3 for speed
            try:
                etf_obj = yf.Ticker(etf)
                info = etf_obj.info
                # Try to get some tickers from related data
                print(f"Checked {etf}")
            except:
                continue

        # If all else fails, return empty to trigger alternative method
        return []

    except Exception as e:
        print(f"Error fetching from ETF: {e}")
        return []

def get_sp500_tickers() -> List[str]:
    """
    Fetch current S&P 500 constituents using multiple methods.
    Returns list of tickers.
    """
    # Try Wikipedia first
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        tables = pd.read_html(response.text)
        print(f"Found {len(tables)} tables on Wikipedia S&P 500 page")

        # Look for the table with the right structure (Symbol + Security columns)
        for i, table in enumerate(tables):
            if 'Symbol' in table.columns and 'Security' in table.columns:
                # This is the constituents table
                tickers = table['Symbol'].astype(str).str.strip().tolist()

                # Replace . with - for tickers (e.g., BRK.B -> BRK-B)
                tickers = [t.replace('.', '-') for t in tickers]

                # Filter out any invalid entries (NaN, empty, etc.)
                tickers = [t for t in tickers if t and t != 'nan' and len(t) <= 6]

                print(f"Found S&P 500 table (table {i}) with {len(tickers)} tickers")

                if len(tickers) >= 400:  # S&P 500 should have ~500 tickers
                    print(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
                    return tickers

        print("Could not find S&P 500 constituents table with expected structure")
    except Exception as e:
        print(f"Error fetching S&P 500 from Wikipedia: {e}")

    # Try ETF method as fallback
    return get_sp500_tickers_from_etf()

def get_nasdaq100_tickers() -> List[str]:
    """
    Fetch current NASDAQ 100 constituents using multiple methods.
    Returns list of tickers.
    """
    # Try Wikipedia first
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        tables = pd.read_html(response.text)
        print(f"Found {len(tables)} tables on Wikipedia NASDAQ-100 page")

        # Look for table with Ticker and Company columns
        for i, table in enumerate(tables):
            if 'Ticker' in table.columns and 'Company' in table.columns:
                # This is likely the constituents table
                tickers = table['Ticker'].astype(str).str.strip().tolist()

                # Replace . with - for tickers
                tickers = [t.replace('.', '-') for t in tickers]

                # Filter out invalid entries
                tickers = [t for t in tickers if t and t != 'nan' and len(t) <= 6]

                print(f"Found NASDAQ-100 table (table {i}) with {len(tickers)} tickers")

                if len(tickers) >= 80:  # NASDAQ 100 should have ~100 tickers
                    print(f"Fetched {len(tickers)} NASDAQ 100 tickers from Wikipedia")
                    return tickers

        print("Could not find NASDAQ 100 constituents table with expected structure")
    except Exception as e:
        print(f"Error fetching NASDAQ 100 from Wikipedia: {e}")

    return []

def get_russell1000_approximation() -> List[str]:
    """
    Approximates Russell 1000 by combining S&P 500, NASDAQ 100,
    and additional major exchange listings.
    """
    sp500 = get_sp500_tickers()
    nasdaq100 = get_nasdaq100_tickers()

    # Combine and deduplicate
    all_tickers = list(set(sp500 + nasdaq100))
    print(f"Combined universe: {len(all_tickers)} unique tickers from major indices")

    if len(all_tickers) < 100:
        print(f"Warning: Only found {len(all_tickers)} tickers, expected more.")
        print("Wikipedia table structure may have changed.")
        print("\nPlease provide a ticker list manually or use an alternative data source.")
        print("Example: Create data/universe/tickers.json with your ticker list")

    return all_tickers


def get_top_n_equities_by_liquidity(
    n: int = 1000,
    save_path: str = "data/universe/",
    lookback_days: int = 60,
    n_workers: int = None,
    use_nasdaq_source: bool = True,
    exclude_etfs: bool = False,
    force_include_tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Get the top N US equities by liquidity (average dollar volume).

    Uses NASDAQ official data for comprehensive coverage, then ranks by actual liquidity.

    Args:
        n: Number of top equities to return
        save_path: Directory to save the universe
        lookback_days: Number of days to compute average volume
        n_workers: Number of parallel workers (None = auto-detect)
        use_nasdaq_source: If True, use NASDAQ official data; else use Wikipedia
        exclude_etfs: If True, removes ETF tickers from ranking universe.
        force_include_tickers: Optional tickers to force into final universe
            (e.g. ["SPY", "QQQ", "IWM"]) for benchmark comparisons.

    Returns:
        DataFrame with top N equities sorted by liquidity
    """
    os.makedirs(save_path, exist_ok=True)

    # Step 1: Get all US symbols
    print("\n=== Step 1: Fetching all US symbols ===")

    listings_df = pd.DataFrame(columns=["ticker", "is_etf"])
    if use_nasdaq_source:
        listings_df = get_all_us_symbols_with_metadata()

        # If NASDAQ fails, fall back to Wikipedia
        if listings_df.empty:
            print("NASDAQ source failed, trying Wikipedia...")
            all_symbols = get_russell1000_approximation()
        else:
            if exclude_etfs:
                listings_df = listings_df[~listings_df["is_etf"]].copy()
            all_symbols = listings_df["ticker"].tolist()
    else:
        all_symbols = get_russell1000_approximation()

    if not all_symbols:
        raise ValueError(
            "Failed to fetch any tickers. Please check:\n"
            "1. Internet connection\n"
            "2. Data sources are accessible\n"
            "3. Try again in a few minutes"
        )

    if exclude_etfs:
        print(f"Found {len(all_symbols)} non-ETF symbols")
    else:
        print(f"Found {len(all_symbols)} total symbols")

    # Step 2: Compute liquidity in parallel
    print("\n=== Step 2: Computing liquidity metrics ===")
    universe_df = compute_liquidity_parallel(
        all_symbols,
        lookback_days=lookback_days,
        n_workers=n_workers
    )

    # Check if we have any data
    if universe_df.empty or len(universe_df) == 0:
        raise ValueError(
            "No tickers successfully fetched liquidity data! This could be due to:\n"
            "1. yfinance API issues\n"
            "2. Network/firewall blocking requests\n"
            "Try running again or check the error messages above."
        )

    # Step 3: Sort by dollar volume and take top N.
    ranked_df = universe_df.sort_values('avg_dollar_volume', ascending=False).copy()
    universe_df = ranked_df.head(n).copy()
    universe_df["forced_include"] = False

    # Optionally force-include benchmark tickers (e.g., SPY/QQQ/IWM).
    if force_include_tickers:
        force_set = {_normalize_ticker(t) for t in force_include_tickers if str(t).strip()}
        if force_set:
            universe_df.loc[universe_df["ticker"].isin(force_set), "forced_include"] = True
            missing_forced = [t for t in sorted(force_set) if t not in set(universe_df["ticker"])]
            if missing_forced:
                forced_rows = ranked_df[ranked_df["ticker"].isin(missing_forced)].copy()
                still_missing = [t for t in missing_forced if t not in set(forced_rows["ticker"])]
                if still_missing:
                    extra = _compute_liquidity_for_tickers(
                        still_missing,
                        lookback_days=lookback_days,
                        min_obs=20,
                    )
                    if not extra.empty:
                        forced_rows = pd.concat([forced_rows, extra], ignore_index=True)
                if not forced_rows.empty:
                    forced_rows["forced_include"] = True
                    universe_df = pd.concat([universe_df, forced_rows], ignore_index=True)
                    universe_df = universe_df.drop_duplicates(subset=["ticker"], keep="first")
            added = universe_df["forced_include"].sum()
            print(f"Forced-included benchmark tickers present: {int(added)}")

    universe_df = universe_df.sort_values("avg_dollar_volume", ascending=False).reset_index(drop=True)
    universe_df['rank'] = range(1, len(universe_df) + 1)

    # Step 4: Save results
    if exclude_etfs:
        csv_path = os.path.join(save_path, f"top_{n}_equities_by_liquidity_non_etf.csv")
        json_path = os.path.join(save_path, f"top_{n}_tickers_non_etf.json")
    else:
        csv_path = os.path.join(save_path, f'top_{n}_equities_by_liquidity.csv')
        json_path = os.path.join(save_path, f'top_{n}_tickers.json')

    universe_df.to_csv(csv_path, index=False)

    # Save just ticker list as JSON for easy importing
    ticker_list = universe_df['ticker'].tolist()
    with open(json_path, 'w') as f:
        json.dump(ticker_list, f, indent=2)

    print(f"\n=== Summary ===")
    if exclude_etfs:
        print(f"Top {n} non-ETF equities by liquidity identified ({len(universe_df)} total incl. forced benchmarks)")
    else:
        print(f"Top {len(universe_df)} equities by liquidity identified")
    print(f"Avg dollar volume range: ${universe_df['avg_dollar_volume'].min():,.0f} - ${universe_df['avg_dollar_volume'].max():,.0f}")
    print(f"Avg volume range: {universe_df['avg_volume'].min():,.0f} - {universe_df['avg_volume'].max():,.0f}")
    print(f"Avg price range: ${universe_df['avg_price'].min():.2f} - ${universe_df['avg_price'].max():.2f}")
    print(f"\nSaved to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")

    print(f"\nTop 10 by liquidity:")
    display_cols = ['rank', 'ticker', 'avg_dollar_volume', 'avg_volume', 'avg_price', 'forced_include']
    display_df = universe_df[display_cols].head(10).copy()
    display_df['avg_dollar_volume'] = display_df['avg_dollar_volume'].apply(lambda x: f"${x:,.0f}")
    display_df['avg_volume'] = display_df['avg_volume'].apply(lambda x: f"{x:,.0f}")
    display_df['avg_price'] = display_df['avg_price'].apply(lambda x: f"${x:.2f}")
    print(display_df.to_string(index=False))

    return universe_df

def load_universe(path: str = "data/universe/top_1000_tickers.json") -> List[str]:
    """
    Load previously saved universe of tickers.

    Args:
        path: Path to JSON file with ticker list

    Returns:
        List of ticker symbols
    """
    with open(path, 'r') as f:
        tickers = json.load(f)
    print(f"Loaded {len(tickers)} tickers from {path}")
    return tickers

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build liquid US equity universe")
    parser.add_argument("--n", type=int, default=1000, help="Number of non-ETF equities to rank")
    parser.add_argument("--lookback_days", type=int, default=60, help="Liquidity lookback window")
    parser.add_argument("--exclude_etfs", action="store_true", help="Exclude ETFs from ranked universe")
    parser.add_argument(
        "--force_include_tickers",
        type=str,
        default="SPY,QQQ,IWM",
        help="Comma-separated benchmark tickers to force-include",
    )
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
        force_include_tickers=force_include,
    )
