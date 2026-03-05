"""
Performance metrics calculations for backtesting.

Extracted and refactored from notebook implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from equity curve.

    Args:
        equity_curve: Series of cumulative portfolio value

    Returns:
        Series of drawdown values (negative percentages)
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown


def calculate_alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> Tuple[float, float, float, float]:
    """
    Calculate alpha and beta via OLS regression.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns

    Returns:
        Tuple of (alpha, beta, alpha_pvalue, beta_pvalue)
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return np.nan, np.nan, np.nan, np.nan

    # Align the series
    aligned = pd.DataFrame({
        'strategy': returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 2:
        return np.nan, np.nan, np.nan, np.nan

    X = sm.add_constant(aligned['benchmark'])
    model = sm.OLS(aligned['strategy'], X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]
    alpha_pval = model.pvalues.iloc[0]
    beta_pval = model.pvalues.iloc[1]

    return alpha, beta, alpha_pval, beta_pval


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Period returns (e.g., monthly)
        benchmark_returns: Optional benchmark returns for alpha/beta
        turnover: Optional turnover series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year (12 for monthly, 252 for daily)

    Returns:
        Dictionary of performance metrics
    """
    metrics = {}

    # Basic return metrics
    mean_return = returns.mean()
    std_return = returns.std()

    metrics['annual_return'] = (1 + mean_return) ** periods_per_year - 1
    metrics['volatility'] = std_return * np.sqrt(periods_per_year)

    # Sharpe ratio
    excess_returns = returns - risk_free_rate / periods_per_year
    if std_return > 0:
        metrics['sharpe_ratio'] = excess_returns.mean() / std_return * np.sqrt(periods_per_year)
    else:
        metrics['sharpe_ratio'] = np.nan

    # CAGR and max drawdown
    equity_curve = (1 + returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1
    years = len(returns) / periods_per_year
    metrics['cagr'] = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    drawdown = calculate_drawdown(equity_curve)
    metrics['max_drawdown'] = drawdown.min()

    # Calmar ratio
    if metrics['max_drawdown'] < 0:
        metrics['calmar_ratio'] = metrics['cagr'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = np.nan

    # Win rate
    metrics['win_rate'] = (returns > 0).mean()

    # Turnover
    if turnover is not None:
        metrics['avg_turnover'] = turnover.mean()

    # Benchmark comparison
    if benchmark_returns is not None:
        alpha, beta, alpha_pval, beta_pval = calculate_alpha_beta(returns, benchmark_returns)
        metrics['alpha'] = alpha * periods_per_year  # Annualize
        metrics['beta'] = beta
        metrics['alpha_pvalue'] = alpha_pval
        metrics['beta_pvalue'] = beta_pval

        # Information ratio
        excess_vs_bench = returns - benchmark_returns
        tracking_error = excess_vs_bench.std() * np.sqrt(periods_per_year)
        if tracking_error > 0:
            metrics['information_ratio'] = excess_vs_bench.mean() / excess_vs_bench.std() * np.sqrt(periods_per_year)
        else:
            metrics['information_ratio'] = np.nan
        metrics['tracking_error'] = tracking_error

    return metrics


def print_metrics_comparison(
    strategy_metrics: Dict[str, float],
    benchmark_metrics: Dict[str, float] = None,
    strategy_name: str = "Strategy",
    benchmark_name: str = "SPY"
) -> None:
    """
    Print side-by-side metrics comparison table.

    Args:
        strategy_metrics: Strategy performance metrics
        benchmark_metrics: Benchmark performance metrics
        strategy_name: Name for strategy column
        benchmark_name: Name for benchmark column
    """
    # Define metrics to display and their formatting
    metrics_display = {
        'annual_return': ('Annual Return', '{:.2%}'),
        'volatility': ('Volatility', '{:.2%}'),
        'sharpe_ratio': ('Sharpe Ratio', '{:.2f}'),
        'max_drawdown': ('Max Drawdown', '{:.2%}'),
        'cagr': ('CAGR', '{:.2%}'),
        'calmar_ratio': ('Calmar Ratio', '{:.2f}'),
        'win_rate': ('Win Rate', '{:.2%}'),
        'avg_turnover': ('Avg Turnover', '{:.2%}'),
        'alpha': ('Alpha (Annual)', '{:.2%}'),
        'beta': ('Beta', '{:.2f}'),
        'information_ratio': ('Information Ratio', '{:.2f}'),
        'tracking_error': ('Tracking Error', '{:.2%}'),
    }

    # Print header
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {strategy_name:>20} {benchmark_name:>20}")
    print("=" * 70)

    # Print each metric
    for key, (label, fmt) in metrics_display.items():
        if key in strategy_metrics:
            strat_val = strategy_metrics[key]
            bench_val = benchmark_metrics.get(key) if benchmark_metrics else None

            # Format strategy value
            if pd.isna(strat_val):
                strat_str = "N/A"
            else:
                strat_str = fmt.format(strat_val)

            # Format benchmark value
            if bench_val is None:
                bench_str = "-"
            elif pd.isna(bench_val):
                bench_str = "N/A"
            else:
                bench_str = fmt.format(bench_val)

            print(f"{label:<25} {strat_str:>20} {bench_str:>20}")

    print("=" * 70 + "\n")
