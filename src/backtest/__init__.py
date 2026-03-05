"""
Backtesting module for trading strategies.

This module provides a production-ready backtesting framework for evaluating
trading strategies on historical data with realistic costs, taxes, and frictions.

Main components:
- Backtester: Core backtesting engine
- Strategy classes: Define stock selection logic
- Metrics: Performance metric calculations
- Plots: Visualization utilities
"""

from .engine import Backtester, BacktestResult
from .strategies import (
    Strategy,
    LongOnlyTopPct,
    LongShortTopBottom,
    RegimeConditionalStrategy,
    CashStrategy,
)
from .metrics import (
    compute_metrics,
    calculate_drawdown,
    calculate_alpha_beta,
    print_metrics_comparison,
)
from .plots import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_turnover,
    plot_rolling_sharpe,
    plot_monthly_returns_heatmap,
    create_backtest_report,
)

__all__ = [
    # Engine
    'Backtester',
    'BacktestResult',
    # Strategies
    'Strategy',
    'LongOnlyTopPct',
    'LongShortTopBottom',
    'RegimeConditionalStrategy',
    'CashStrategy',
    # Metrics
    'compute_metrics',
    'calculate_drawdown',
    'calculate_alpha_beta',
    'print_metrics_comparison',
    # Plots
    'plot_cumulative_returns',
    'plot_drawdown',
    'plot_turnover',
    'plot_rolling_sharpe',
    'plot_monthly_returns_heatmap',
    'create_backtest_report',
]
