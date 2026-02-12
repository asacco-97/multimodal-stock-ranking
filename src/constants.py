# src/constants.py
"""
Data dictionary for the stock ranking pipeline.
Maps variable names to their descriptions and metadata.
"""

# ============================================================================
# DATA DICTIONARY
# ============================================================================

DATA_DICTIONARY = {
    # ------------------------------------------------------------------------
    # Identifiers and Metadata
    # ------------------------------------------------------------------------
    'ticker': 'Stock ticker symbol (e.g., AAPL, MSFT)',
    'date': 'Trading date in YYYY-MM-DD format',
    'fetch_date': 'Date when the data was fetched from the API',
    'sector': 'GICS sector classification (e.g., Technology, Healthcare)',
    'industry': 'GICS industry classification (e.g., Software, Pharmaceuticals)',

    # ------------------------------------------------------------------------
    # Price and Volume (OHLCV)
    # ------------------------------------------------------------------------
    'open': 'Opening price for the trading day',
    'high': 'Highest price during the trading day',
    'low': 'Lowest price during the trading day',
    'close': 'Closing price for the trading day (adjusted for splits/dividends)',
    'volume': 'Number of shares traded during the day',

    # ------------------------------------------------------------------------
    # Returns and Forward-Looking Targets
    # ------------------------------------------------------------------------
    'return_t+1': 'Forward 1-day return: (close_t+1 - close_t) / close_t',
    'return_t+5': 'Forward 5-day return: (close_t+5 - close_t) / close_t',
    'return_t+20': 'Forward 20-day return (4 weeks): (close_t+20 - close_t) / close_t',
    'return_t+60': 'Forward 60-day return (3 months): (close_t+60 - close_t) / close_t',

    # Target variables (created from forward returns)
    'target_1d': 'Target variable based on 1-day forward returns (quintile/decile ranking)',
    'target_5d': 'Target variable based on 5-day forward returns (quintile/decile ranking)',
    'target_20d': 'Target variable based on 20-day forward returns (quintile/decile ranking)',
    'target_60d': 'Target variable based on 60-day forward returns (quintile/decile ranking)',

    # ------------------------------------------------------------------------
    # Technical Indicators - Momentum
    # ------------------------------------------------------------------------
    'momentum_5': '5-day momentum: (close - close_5d_ago) / close_5d_ago',
    'momentum_10': '10-day momentum: (close - close_10d_ago) / close_10d_ago',
    'momentum_20': '20-day momentum: (close - close_20d_ago) / close_20d_ago',
    'momentum_60d': '60-day momentum: (close - close_60d_ago) / close_60d_ago',
    'momentum_120d': '120-day momentum: (close - close_120d_ago) / close_120d_ago',

    # ------------------------------------------------------------------------
    # Technical Indicators - Volatility
    # ------------------------------------------------------------------------
    'volatility_5': '5-day rolling standard deviation of returns',
    'volatility_10': '10-day rolling standard deviation of returns',
    'volatility_20': '20-day rolling standard deviation of returns',
    'volatility_60d': '60-day rolling standard deviation of returns',

    # ------------------------------------------------------------------------
    # Technical Indicators - Moving Averages
    # ------------------------------------------------------------------------
    'sma_5': '5-day simple moving average of closing price',
    'sma_10': '10-day simple moving average of closing price',
    'sma_20': '20-day simple moving average of closing price',

    # ------------------------------------------------------------------------
    # Technical Indicators - Oscillators and Drawdown
    # ------------------------------------------------------------------------
    'rsi_14': '14-day Relative Strength Index (0-100 scale, overbought > 70, oversold < 30)',
    'rsi_14d': '14-day Relative Strength Index (0-100 scale, overbought > 70, oversold < 30)',
    'cumulative_return': 'Cumulative return from the start of the time series',
    'running_max': 'Running maximum of cumulative returns (used for drawdown calculation)',
    'drawdown': 'Current drawdown from running maximum (negative value)',
    'max_drawdown_60d': 'Maximum drawdown over the past 60 days',

    # ------------------------------------------------------------------------
    # Fundamental - Valuation Ratios
    # ------------------------------------------------------------------------
    'market_cap': 'Market capitalization: share price × shares outstanding',
    'enterprise_value': 'Enterprise value: market cap + total debt - total cash',
    'trailing_pe': 'Trailing P/E ratio: price / earnings per share (last 12 months)',
    'forward_pe': 'Forward P/E ratio: price / estimated forward earnings per share',
    'pe_ratio': 'Price-to-Earnings ratio: price / earnings per share',
    'peg_ratio': 'PEG ratio: P/E ratio / earnings growth rate (value < 1 often considered attractive)',
    'price_to_book': 'Price-to-Book ratio: market cap / book value',
    'pb_ratio': 'Price-to-Book ratio: market cap / book value',
    'price_to_sales': 'Price-to-Sales ratio: market cap / total revenue',
    'ps_ratio': 'Price-to-Sales ratio: market cap / total revenue',
    'enterprise_to_revenue': 'Enterprise Value-to-Revenue ratio: EV / total revenue',
    'enterprise_to_ebitda': 'Enterprise Value-to-EBITDA ratio: EV / EBITDA',

    # ------------------------------------------------------------------------
    # Fundamental - Profitability Metrics
    # ------------------------------------------------------------------------
    'profit_margin': 'Net profit margin: net income / revenue (as decimal, e.g., 0.15 = 15%)',
    'operating_margin': 'Operating margin: operating income / revenue',
    'gross_margin': 'Gross margin: (revenue - COGS) / revenue',
    'roe': 'Return on Equity: net income / shareholders equity',
    'roa': 'Return on Assets: net income / total assets',

    # ------------------------------------------------------------------------
    # Fundamental - Growth Metrics
    # ------------------------------------------------------------------------
    'revenue_growth': 'Year-over-year revenue growth rate (as decimal)',
    'earnings_growth': 'Year-over-year earnings growth rate (as decimal)',

    # ------------------------------------------------------------------------
    # Fundamental - Financial Health
    # ------------------------------------------------------------------------
    'total_cash': 'Total cash and cash equivalents on balance sheet',
    'total_debt': 'Total debt (short-term + long-term)',
    'debt_to_equity': 'Debt-to-Equity ratio: total debt / shareholders equity',
    'current_ratio': 'Current ratio: current assets / current liabilities (liquidity measure)',
    'quick_ratio': 'Quick ratio: (current assets - inventory) / current liabilities',
    'book_value': 'Book value (shareholders equity)',

    # ------------------------------------------------------------------------
    # Fundamental - Per-Share Metrics
    # ------------------------------------------------------------------------
    'revenue_per_share': 'Total revenue / shares outstanding',
    'earnings_per_share': 'Net income / shares outstanding (EPS)',

    # ------------------------------------------------------------------------
    # Fundamental - Dividends
    # ------------------------------------------------------------------------
    'dividend_rate': 'Annual dividend per share (in dollars)',
    'dividend_yield': 'Annual dividend yield: dividend rate / share price',
    'payout_ratio': 'Dividend payout ratio: dividends / earnings',

    # ------------------------------------------------------------------------
    # Fundamental - Share Structure and Ownership
    # ------------------------------------------------------------------------
    'beta': 'Beta coefficient: volatility relative to the market (SPY)',
    'shares_outstanding': 'Total number of shares outstanding',
    'float_shares': 'Number of shares available for public trading (excludes restricted shares)',
    'held_percent_insiders': 'Percentage of shares held by company insiders',
    'held_percent_institutions': 'Percentage of shares held by institutional investors',

    # ------------------------------------------------------------------------
    # Macroeconomic - Interest Rates and Monetary Policy
    # ------------------------------------------------------------------------
    'DFF': 'Federal Funds Effective Rate (FRED series DFF) - overnight interbank lending rate',
    'DGS10': '10-Year Treasury Constant Maturity Rate (FRED series DGS10)',
    'DGS2': '2-Year Treasury Constant Maturity Rate (FRED series DGS2)',
    'T10Y2Y': '10-Year minus 2-Year Treasury Spread (FRED series T10Y2Y) - recession indicator when inverted',

    # ------------------------------------------------------------------------
    # Macroeconomic - Inflation
    # ------------------------------------------------------------------------
    'CPIAUCSL': 'Consumer Price Index for All Urban Consumers (FRED series CPIAUCSL)',
    'CPILFESL': 'Core CPI excluding food and energy (FRED series CPILFESL)',
    'PCEPI': 'Personal Consumption Expenditures Price Index (FRED series PCEPI) - Fed preferred inflation measure',

    # ------------------------------------------------------------------------
    # Macroeconomic - Economic Growth
    # ------------------------------------------------------------------------
    'GDP': 'Gross Domestic Product (FRED series GDP) - nominal GDP in billions',
    'GDPC1': 'Real Gross Domestic Product (FRED series GDPC1) - inflation-adjusted GDP',
    'INDPRO': 'Industrial Production Index (FRED series INDPRO) - manufacturing and production output',

    # ------------------------------------------------------------------------
    # Macroeconomic - Employment
    # ------------------------------------------------------------------------
    'UNRATE': 'Unemployment Rate (FRED series UNRATE) - civilian unemployment rate as percentage',
    'PAYEMS': 'All Employees: Total Nonfarm Payrolls (FRED series PAYEMS) - monthly job additions',
    'ICSA': 'Initial Claims for Unemployment Insurance (FRED series ICSA) - weekly jobless claims',

    # ------------------------------------------------------------------------
    # Macroeconomic - Consumer and Business Sentiment
    # ------------------------------------------------------------------------
    'UMCSENT': 'University of Michigan Consumer Sentiment Index (FRED series UMCSENT)',
    'RSXFS': 'Advance Retail Sales (FRED series RSXFS) - monthly retail sales excluding food services',

    # ------------------------------------------------------------------------
    # Macroeconomic - Market Indicators
    # ------------------------------------------------------------------------
    'VIXCLS': 'CBOE Volatility Index (VIX) - implied volatility of S&P 500 options (fear gauge)',
    'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate (WTI) (FRED series DCOILWTICO)',

    # ------------------------------------------------------------------------
    # Macroeconomic - Money Supply and Credit
    # ------------------------------------------------------------------------
    'M2SL': 'M2 Money Stock (FRED series M2SL) - broad measure of money supply',
    'TOTALSL': 'Total Consumer Credit Outstanding (FRED series TOTALSL)',

    # ------------------------------------------------------------------------
    # Cross-Sectional Features - Percentile Ranks
    # ------------------------------------------------------------------------
    # These are created by add_percentile_ranks() with suffix '_rank'
    # Examples: momentum_60d_rank, pe_ratio_rank, roe_rank, etc.
    # Pattern: {feature}_rank = Percentile rank (0-1) of {feature} within each date

    # ------------------------------------------------------------------------
    # Cross-Sectional Features - Z-Scores
    # ------------------------------------------------------------------------
    # These are created by add_z_scores() with suffix '_zscore'
    # Examples: momentum_60d_zscore, pe_ratio_zscore, roe_zscore, etc.
    # Pattern: {feature}_zscore = Z-score of {feature} within each date (mean=0, std=1)

    # ------------------------------------------------------------------------
    # Cross-Sectional Features - Sector-Relative
    # ------------------------------------------------------------------------
    # These are created by add_sector_relative_features() with suffix '_sector_rel'
    # Examples: momentum_60d_sector_rel, pe_ratio_sector_rel, roe_sector_rel, etc.
    # Pattern: {feature}_sector_rel = {feature} - sector_mean({feature})

    # ------------------------------------------------------------------------
    # Cross-Sectional Features - Sector Ranks
    # ------------------------------------------------------------------------
    # These are created by add_sector_ranks() with suffix '_sector_rank'
    # Examples: momentum_60d_sector_rank, pe_ratio_sector_rank, roe_sector_rank, etc.
    # Pattern: {feature}_sector_rank = Percentile rank (0-1) of {feature} within sector and date

    # ------------------------------------------------------------------------
    # Cross-Sectional Features - Quintile Buckets
    # ------------------------------------------------------------------------
    # These are created by add_quintile_buckets() with suffix '_quintile'
    # Examples: momentum_60d_quintile, rsi_14d_quintile, pe_ratio_quintile, etc.
    # Pattern: {feature}_quintile = Quintile bucket (0-4) of {feature} within each date

    # ------------------------------------------------------------------------
    # Cross-Sectional Features - Rank Momentum
    # ------------------------------------------------------------------------
    # These are created by add_rank_changes() with suffix '_rank_change_Xd'
    # Examples: momentum_60d_rank_change_1d, momentum_60d_rank_change_5d, etc.
    # Pattern: {feature}_rank_change_Xd = Change in rank over X days
}


# ============================================================================
# FEATURE CATEGORIES
# ============================================================================

FEATURE_CATEGORIES = {
    'identifiers': ['ticker', 'date', 'fetch_date', 'sector', 'industry'],

    'price_volume': ['open', 'high', 'low', 'close', 'volume'],

    'returns': ['return_t+1', 'return_t+5', 'return_t+20', 'return_t+60'],

    'targets': ['target_1d', 'target_5d', 'target_20d', 'target_60d'],

    'technical_momentum': [
        'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60d', 'momentum_120d',
        'cumulative_return'
    ],

    'technical_volatility': [
        'volatility_5', 'volatility_10', 'volatility_20', 'volatility_60d',
        'drawdown', 'running_max', 'max_drawdown_60d'
    ],

    'technical_indicators': [
        'sma_5', 'sma_10', 'sma_20',
        'rsi_14', 'rsi_14d'
    ],

    'fundamental_valuation': [
        'market_cap', 'enterprise_value',
        'trailing_pe', 'forward_pe', 'pe_ratio', 'peg_ratio',
        'price_to_book', 'pb_ratio', 'price_to_sales', 'ps_ratio',
        'enterprise_to_revenue', 'enterprise_to_ebitda'
    ],

    'fundamental_profitability': [
        'profit_margin', 'operating_margin', 'gross_margin',
        'roe', 'roa'
    ],

    'fundamental_growth': [
        'revenue_growth', 'earnings_growth'
    ],

    'fundamental_financial_health': [
        'total_cash', 'total_debt', 'debt_to_equity',
        'current_ratio', 'quick_ratio', 'book_value'
    ],

    'fundamental_per_share': [
        'revenue_per_share', 'earnings_per_share'
    ],

    'fundamental_dividends': [
        'dividend_rate', 'dividend_yield', 'payout_ratio'
    ],

    'fundamental_ownership': [
        'beta', 'shares_outstanding', 'float_shares',
        'held_percent_insiders', 'held_percent_institutions'
    ],

    'macro_interest_rates': [
        'DFF', 'DGS10', 'DGS2', 'T10Y2Y'
    ],

    'macro_inflation': [
        'CPIAUCSL', 'CPILFESL', 'PCEPI'
    ],

    'macro_growth': [
        'GDP', 'GDPC1', 'INDPRO'
    ],

    'macro_employment': [
        'UNRATE', 'PAYEMS', 'ICSA'
    ],

    'macro_sentiment': [
        'UMCSENT', 'RSXFS'
    ],

    'macro_market': [
        'VIXCLS', 'DCOILWTICO'
    ],

    'macro_money_credit': [
        'M2SL', 'TOTALSL'
    ]
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_description(variable_name: str) -> str:
    """
    Get description for a variable.

    Args:
        variable_name: Name of the variable

    Returns:
        Description string, or generated description for pattern-based features
    """
    # Check if exact match exists
    if variable_name in DATA_DICTIONARY:
        return DATA_DICTIONARY[variable_name]

    # Handle pattern-based cross-sectional features
    if variable_name.endswith('_rank'):
        base_feature = variable_name[:-5]
        return f"Percentile rank (0-1) of {base_feature} within each date"

    elif variable_name.endswith('_zscore'):
        base_feature = variable_name[:-7]
        return f"Z-score of {base_feature} within each date (mean=0, std=1)"

    elif variable_name.endswith('_sector_rel'):
        base_feature = variable_name[:-11]
        return f"{base_feature} minus sector mean within each date"

    elif variable_name.endswith('_sector_rank'):
        base_feature = variable_name[:-12]
        return f"Percentile rank (0-1) of {base_feature} within sector and date"

    elif variable_name.endswith('_quintile'):
        base_feature = variable_name[:-9]
        return f"Quintile bucket (0-4) of {base_feature} within each date"

    elif '_rank_change_' in variable_name:
        parts = variable_name.split('_rank_change_')
        base_feature = parts[0]
        period = parts[1] if len(parts) > 1 else 'X'
        return f"Change in rank of {base_feature} over {period}"

    elif variable_name.startswith('return_t+'):
        days = variable_name.split('+')[1]
        return f"Forward {days}-day return: (close_t+{days} - close_t) / close_t"

    elif variable_name.startswith('target_') and variable_name.endswith('d'):
        days = variable_name[7:-1]  # Extract number between 'target_' and 'd'
        return f"Target variable based on {days}-day forward returns (quintile/decile ranking)"

    return 'No description available'


def get_category(variable_name: str) -> str:
    """
    Get category for a variable.

    Args:
        variable_name: Name of the variable

    Returns:
        Category name, or 'cross_sectional' for derived features, or 'unknown' if not found
    """
    for category, variables in FEATURE_CATEGORIES.items():
        if variable_name in variables:
            return category

    # Check for cross-sectional features by suffix
    suffixes = ['_rank', '_zscore', '_sector_rel', '_sector_rank', '_quintile', '_rank_change_']
    for suffix in suffixes:
        if suffix in variable_name:
            return 'cross_sectional'

    # Check for dynamic return/target columns
    if variable_name.startswith('return_t+') or (variable_name.startswith('target_') and variable_name.endswith('d')):
        return 'targets' if variable_name.startswith('target_') else 'returns'

    return 'unknown'


def get_variables_by_category(category: str) -> list:
    """
    Get list of variables in a category.

    Args:
        category: Category name

    Returns:
        List of variable names in that category
    """
    return FEATURE_CATEGORIES.get(category, [])


def print_data_dictionary(categories: list = None):
    """
    Print the data dictionary in a formatted way.

    Args:
        categories: List of categories to print (None = all)
    """
    if categories is None:
        categories = FEATURE_CATEGORIES.keys()

    for category in categories:
        if category not in FEATURE_CATEGORIES:
            continue

        print(f"\n{'=' * 80}")
        print(f"{category.upper().replace('_', ' ')}")
        print('=' * 80)

        variables = FEATURE_CATEGORIES[category]
        for var in variables:
            desc = get_description(var)
            print(f"\n{var:30} {desc}")


if __name__ == "__main__":
    # Example usage
    print("Stock Ranking Pipeline - Data Dictionary")
    print("=" * 80)

    # Print all categories
    print_data_dictionary()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total base variables: {len(DATA_DICTIONARY)}")
    print(f"Total categories: {len(FEATURE_CATEGORIES)}")

    print("\nVariables per category:")
    for category, variables in FEATURE_CATEGORIES.items():
        print(f"  {category:30} {len(variables):3} variables")

    print("\n" + "=" * 80)
    print("CROSS-SECTIONAL FEATURES (Pattern-Based)")
    print("=" * 80)
    print("\nThe following suffixes create derived features:")
    print("  *_rank          : Percentile rank (0-1) within each date")
    print("  *_zscore        : Z-score (mean=0, std=1) within each date")
    print("  *_sector_rel    : Value minus sector mean within each date")
    print("  *_sector_rank   : Percentile rank within sector and date")
    print("  *_quintile      : Quintile bucket (0-4) within each date")
    print("  *_rank_change_Xd: Change in rank over X days")

    print("\n" + "=" * 80)
    print("EXAMPLE LOOKUPS")
    print("=" * 80)

    # Test pattern-based lookups
    test_vars = [
        'momentum_60d_rank',
        'pe_ratio_zscore',
        'roe_sector_rel',
        'return_t+20',
        'target_20d'
    ]

    for var in test_vars:
        desc = get_description(var)
        category = get_category(var)
        print(f"\n{var:30}")
        print(f"  Category: {category}")
        print(f"  Description: {desc}")
