"""
Visualization utilities for backtesting results.

Creates publication-quality plots for backtest analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
from .engine import BacktestResult
from .metrics import calculate_drawdown


def plot_cumulative_returns(
    result: BacktestResult,
    benchmark_name: str = "SPY",
    figsize: Tuple[int, int] = (14, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot cumulative returns: Strategy vs Benchmark.

    Args:
        result: BacktestResult from backtester
        benchmark_name: Name of benchmark for legend
        figsize: Figure size (width, height)
        title: Optional custom title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate cumulative returns
    strategy_cum = (1 + result.returns).cumprod()
    benchmark_cum = (1 + result.benchmark_returns).cumprod()

    # Plot
    ax.plot(strategy_cum.index, strategy_cum.values,
            label='Strategy', linewidth=2.5, color='#2E86AB')
    ax.plot(benchmark_cum.index, benchmark_cum.values,
            label=benchmark_name, linewidth=2, color='#A23B72', linestyle='--', alpha=0.8)

    # Formatting
    ax.set_ylabel('Cumulative Return (Multiple)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title(title or f'Cumulative Returns: Strategy vs {benchmark_name}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)

    # Add final values as text
    final_strat = strategy_cum.iloc[-1]
    final_bench = benchmark_cum.iloc[-1]
    ax.text(0.98, 0.02,
            f'Final: Strategy {final_strat:.2f}x | {benchmark_name} {final_bench:.2f}x',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)

    plt.tight_layout()
    return fig


def plot_drawdown(
    result: BacktestResult,
    benchmark_name: str = "SPY",
    figsize: Tuple[int, int] = (14, 5),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot drawdown over time: Strategy vs Benchmark.

    Args:
        result: BacktestResult from backtester
        benchmark_name: Name of benchmark for legend
        figsize: Figure size (width, height)
        title: Optional custom title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate drawdowns
    strategy_dd = calculate_drawdown((1 + result.returns).cumprod())
    benchmark_dd = calculate_drawdown((1 + result.benchmark_returns).cumprod())

    # Plot
    ax.fill_between(strategy_dd.index, strategy_dd.values * 100, 0,
                     alpha=0.6, color='#E63946', label='Strategy')
    ax.fill_between(benchmark_dd.index, benchmark_dd.values * 100, 0,
                     alpha=0.4, color='#457B9D', label=benchmark_name)

    # Formatting
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title(title or f'Drawdown: Strategy vs {benchmark_name}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')

    # Add max drawdown annotations
    max_strat_dd = strategy_dd.min() * 100
    max_bench_dd = benchmark_dd.min() * 100
    ax.text(0.98, 0.98,
            f'Max DD: Strategy {max_strat_dd:.1f}% | {benchmark_name} {max_bench_dd:.1f}%',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)

    plt.tight_layout()
    return fig


def plot_turnover(
    result: BacktestResult,
    figsize: Tuple[int, int] = (14, 4),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot portfolio turnover over time.

    Args:
        result: BacktestResult from backtester
        figsize: Figure size (width, height)
        title: Optional custom title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Turnover is already normalized in engine:
    # min(purchases, sales) / average portfolio value.
    normalized_turnover = result.turnover.values

    # Plot
    ax.bar(result.turnover.index, normalized_turnover * 100,
           color='#F77F00', alpha=0.7, width=20)

    # Formatting
    ax.set_ylabel('Turnover (% of Portfolio)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Portfolio Turnover',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Add average turnover
    avg_turnover = normalized_turnover.mean() * 100
    ax.axhline(y=avg_turnover, color='red', linestyle='--', linewidth=2, alpha=0.6,
               label=f'Avg: {avg_turnover:.1f}%')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)

    plt.tight_layout()
    return fig


def plot_rolling_sharpe(
    result: BacktestResult,
    window: int = 12,
    figsize: Tuple[int, int] = (14, 5),
    title: Optional[str] = None,
    benchmark_name: str = "SPY"
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio over time.

    Args:
        result: BacktestResult from backtester
        window: Rolling window size (number of periods)
        figsize: Figure size (width, height)
        title: Optional custom title
        benchmark_name: Name of benchmark for legend

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate rolling Sharpe
    rolling_mean = result.returns.rolling(window).mean()
    rolling_std = result.returns.rolling(window).std()

    # Annualization factor (assuming monthly data if window=12)
    periods_per_year = 12 if window == 12 else 252
    sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)

    # Same for benchmark
    bench_rolling_mean = result.benchmark_returns.rolling(window).mean()
    bench_rolling_std = result.benchmark_returns.rolling(window).std()
    bench_sharpe = (bench_rolling_mean / bench_rolling_std) * np.sqrt(periods_per_year)

    # Plot
    ax.plot(sharpe.index, sharpe.values,
            label='Strategy', linewidth=2.5, color='#2E86AB')
    ax.plot(bench_sharpe.index, bench_sharpe.values,
            label=benchmark_name, linewidth=2, color='#A23B72', linestyle='--', alpha=0.8)

    # Formatting
    ax.set_ylabel(f'{window}-Period Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title(title or f'Rolling Sharpe Ratio ({window}-period window)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax.axhline(y=1.0, color='green', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=2.0, color='green', linestyle=':', linewidth=1, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
    result: BacktestResult,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot monthly returns as a heatmap (years x months).

    Args:
        result: BacktestResult from backtester
        figsize: Figure size (width, height)
        title: Optional custom title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to monthly if not already
    returns_series = result.returns.copy()
    returns_series.index = pd.to_datetime(returns_series.index)

    # Resample to monthly (in case it's daily)
    monthly_returns = returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # Create year-month pivot
    df = pd.DataFrame({
        'return': monthly_returns.values * 100,  # Convert to percentage
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month
    })

    # Pivot table
    pivot = df.pivot(index='year', columns='month', values='return')

    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = [month_names[m-1] for m in pivot.columns]

    # Plot heatmap
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Return (%)'}, ax=ax,
                linewidths=0.5, linecolor='gray')

    # Formatting
    ax.set_ylabel('Year', fontsize=12, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Monthly Returns Heatmap (%)',
                 fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    return fig


def create_backtest_report(
    result: BacktestResult,
    benchmark_name: str = "SPY",
    save_path: Optional[str] = None,
    strategy_name: str = "Strategy"
) -> plt.Figure:
    """
    Generate comprehensive backtest report with all plots and metrics.

    Creates a multi-panel figure with:
    - Cumulative returns
    - Drawdown
    - Rolling Sharpe
    - Monthly returns heatmap
    - Turnover
    - Metrics table

    Args:
        result: BacktestResult from backtester
        benchmark_name: Name of benchmark (e.g., 'SPY', 'QQQ')
        save_path: Optional path to save figure (e.g., 'report.png')
        strategy_name: Name of strategy for title

    Returns:
        matplotlib Figure object
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(6, 2, hspace=0.35, wspace=0.3,
                          top=0.95, bottom=0.05, left=0.08, right=0.95)

    # Main title
    fig.suptitle(f'Backtest Report: {strategy_name}',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Cumulative Returns (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    strategy_cum = (1 + result.returns).cumprod()
    benchmark_cum = (1 + result.benchmark_returns).cumprod()
    ax1.plot(strategy_cum.index, strategy_cum.values,
             label=strategy_name, linewidth=2.5, color='#2E86AB')
    ax1.plot(benchmark_cum.index, benchmark_cum.values,
             label=benchmark_name, linewidth=2, color='#A23B72', linestyle='--', alpha=0.8)
    ax1.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
    ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.axhline(y=1.0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)

    # 2. Drawdown (second row, full width)
    ax2 = fig.add_subplot(gs[1, :])
    strategy_dd = calculate_drawdown(strategy_cum)
    benchmark_dd = calculate_drawdown(benchmark_cum)
    ax2.fill_between(strategy_dd.index, strategy_dd.values * 100, 0,
                      alpha=0.6, color='#E63946', label=strategy_name)
    ax2.fill_between(benchmark_dd.index, benchmark_dd.values * 100, 0,
                      alpha=0.4, color='#457B9D', label=benchmark_name)
    ax2.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Drawdown', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=10, loc='lower left')
    ax2.grid(True, alpha=0.3, linestyle=':')

    # 3. Rolling Sharpe (third row, left)
    ax3 = fig.add_subplot(gs[2, 0])
    window = 12
    rolling_mean = result.returns.rolling(window).mean()
    rolling_std = result.returns.rolling(window).std()
    sharpe = (rolling_mean / rolling_std) * np.sqrt(12)
    bench_rolling_mean = result.benchmark_returns.rolling(window).mean()
    bench_rolling_std = result.benchmark_returns.rolling(window).std()
    bench_sharpe = (bench_rolling_mean / bench_rolling_std) * np.sqrt(12)
    ax3.plot(sharpe.index, sharpe.values, label=strategy_name, linewidth=2, color='#2E86AB')
    ax3.plot(bench_sharpe.index, bench_sharpe.values,
             label=benchmark_name, linewidth=1.5, color='#A23B72', linestyle='--', alpha=0.8)
    ax3.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax3.set_title('Rolling 12-Month Sharpe', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)

    # 4. Turnover (third row, right)
    ax4 = fig.add_subplot(gs[2, 1])
    normalized_turnover = result.turnover.values
    ax4.bar(result.turnover.index, normalized_turnover * 100,
            color='#F77F00', alpha=0.7, width=20)
    avg_turnover = normalized_turnover.mean() * 100
    ax4.axhline(y=avg_turnover, color='red', linestyle='--', linewidth=2, alpha=0.6,
                label=f'Avg: {avg_turnover:.1f}%')
    ax4.set_ylabel('Turnover (% Portfolio)', fontsize=11, fontweight='bold')
    ax4.set_title('Portfolio Turnover', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y', linestyle=':')

    # 5. Monthly Returns Heatmap (fourth and fifth rows, full width)
    ax5 = fig.add_subplot(gs[3:5, :])
    returns_series = result.returns.copy()
    returns_series.index = pd.to_datetime(returns_series.index)
    monthly_returns = returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    df = pd.DataFrame({
        'return': monthly_returns.values * 100,
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month
    })
    pivot = df.pivot(index='year', columns='month', values='return')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = [month_names[m-1] for m in pivot.columns]
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Return (%)'}, ax=ax5,
                linewidths=0.5, linecolor='gray', annot_kws={'fontsize': 8})
    ax5.set_ylabel('Year', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax5.set_title('Monthly Returns (%)', fontsize=12, fontweight='bold', pad=10)

    # 6. Metrics Table (bottom row, full width)
    ax6 = fig.add_subplot(gs[5, :])
    ax6.axis('off')

    # Format metrics table
    metrics_data = []
    metric_labels = {
        'annual_return': 'Annual Return',
        'volatility': 'Volatility',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
        'cagr': 'CAGR',
        'calmar_ratio': 'Calmar Ratio',
        'win_rate': 'Win Rate',
        'alpha': 'Alpha (Annual)',
        'beta': 'Beta',
        'information_ratio': 'Info Ratio',
    }

    for key, label in metric_labels.items():
        if key in result.metrics:
            strat_val = result.metrics[key]
            bench_val = result.benchmark_metrics.get(key, np.nan)

            # Format values
            if key in ['annual_return', 'volatility', 'max_drawdown', 'cagr', 'win_rate', 'alpha']:
                strat_str = f'{strat_val:.2%}' if not pd.isna(strat_val) else 'N/A'
                bench_str = f'{bench_val:.2%}' if not pd.isna(bench_val) else 'N/A'
            else:
                strat_str = f'{strat_val:.2f}' if not pd.isna(strat_val) else 'N/A'
                bench_str = f'{bench_val:.2f}' if not pd.isna(bench_val) else 'N/A'

            metrics_data.append([label, strat_str, bench_str])

    # Create table
    table = ax6.table(cellText=metrics_data,
                      colLabels=['Metric', strategy_name, benchmark_name],
                      cellLoc='center',
                      loc='center',
                      bbox=[0.1, 0.0, 0.8, 1.0])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(metrics_data) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Report saved to {save_path}")

    return fig
