"""
Core backtesting engine.

Extracted and refactored from notebook implementations to provide a
production-ready, reusable backtesting framework.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .strategies import Strategy
from .metrics import compute_metrics


def _safe_price_return(start_price: float, end_price: float) -> float:
    start = pd.to_numeric(start_price, errors="coerce")
    end = pd.to_numeric(end_price, errors="coerce")
    if pd.isna(start) or pd.isna(end) or start <= 0:
        return 0.0
    return float(end / start - 1.0)


@dataclass
class BacktestResult:
    """Container for backtesting results."""

    equity_curve: pd.Series          # Indexed by date, cumulative equity
    returns: pd.Series               # Period returns
    positions: pd.DataFrame          # Columns: date, ticker, weight, shares, position_value
    turnover: pd.Series              # Turnover per rebalance period
    metrics: Dict[str, float]        # Performance metrics
    benchmark_equity: pd.Series      # Benchmark equity curve
    benchmark_returns: pd.Series     # Benchmark returns
    benchmark_metrics: Dict[str, float]  # Benchmark metrics
    cash_flows: Optional[pd.Series] = None  # External contributions/withdrawals


class Backtester:
    """
    Main backtesting engine with realistic costs, taxes, and frictions.

    Supports both monthly and daily rebalancing with configurable strategies.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 100000,
        transaction_cost_bps: float = 10,
        short_term_tax_rate: float = 0.35,
        borrow_cost_annual: float = 0.03,
        rebalance_frequency: str = 'monthly',
        price_col: str = 'close',
        date_col: str = 'date',
        ticker_col: str = 'ticker',
        benchmark_ticker: str = 'SPY',
        missing_price_mode: str = 'carry',
        max_abs_ticker_return: Optional[float] = 5.0,
        liquidation_floor: float = -0.95,
        periodic_contribution: float = 0.0,
        contribution_schedule: Optional[Dict] = None,
        buying_power_multiplier: float = 1.0,
    ):
        """
        Args:
            data: DataFrame with columns [date, ticker, close, pred, ...]
            strategy: Strategy instance defining selection logic
            initial_capital: Starting capital
            transaction_cost_bps: Trading costs in basis points
            short_term_tax_rate: Short-term capital gains tax rate
            borrow_cost_annual: Annual borrow cost for shorts
            rebalance_frequency: 'monthly' or 'daily'
            price_col: Column name for prices
            date_col: Column name for dates
            ticker_col: Column name for tickers
            benchmark_ticker: Ticker to use for benchmark (e.g., 'SPY', 'QQQ')
            missing_price_mode: How to handle missing end prices for held names.
                'carry' (default): assume end_price == start_price (0 return).
                'worst': long -> 0, short -> 2x start price (legacy pessimistic).
            max_abs_ticker_return: Optional per-ticker close-to-close return clip
                (e.g., 5.0 = +/-500%) to guard against bad prints/split artifacts.
            liquidation_floor: Minimum allowed period return to avoid negative-equity
                cascades from a single period (e.g., -0.95).
            periodic_contribution: Fixed cash flow applied each rebalance date
                before trades (negative values withdraw capital).
            contribution_schedule: Optional mapping of date -> cash flow applied
                on matching rebalance dates (in addition to periodic_contribution).
            buying_power_multiplier: Position sizing multiplier on equity.
                1.0 = no leverage, 1.5 = 150% gross buying power, etc.
        """
        if date_col not in data.columns:
            raise ValueError(f"date_col '{date_col}' not found in data")
        if ticker_col not in data.columns:
            raise ValueError(f"ticker_col '{ticker_col}' not found in data")

        self.data = data.copy()
        self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
        self.data = self.data.dropna(subset=[date_col]).sort_values(date_col)

        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.short_term_tax_rate = short_term_tax_rate
        self.borrow_cost_annual = borrow_cost_annual
        self.rebalance_frequency = rebalance_frequency
        self.price_col = price_col
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.benchmark_ticker = benchmark_ticker
        self.missing_price_mode = missing_price_mode
        self.max_abs_ticker_return = max_abs_ticker_return
        self.liquidation_floor = liquidation_floor
        self.periodic_contribution = float(periodic_contribution)
        self.buying_power_multiplier = float(buying_power_multiplier)
        self.contribution_schedule = self._normalize_contribution_schedule(contribution_schedule)

        if self.price_col not in self.data.columns:
            raise ValueError(f"price_col '{self.price_col}' not found in data")
        if self.missing_price_mode not in {"carry", "worst"}:
            raise ValueError("missing_price_mode must be one of {'carry', 'worst'}")
        if self.max_abs_ticker_return is not None and self.max_abs_ticker_return <= 0:
            raise ValueError("max_abs_ticker_return must be > 0 or None")
        if self.liquidation_floor <= -1.0 or self.liquidation_floor >= 0.0:
            raise ValueError("liquidation_floor must be in (-1.0, 0.0)")
        if self.buying_power_multiplier <= 0:
            raise ValueError("buying_power_multiplier must be > 0")

        # Keep strategy and engine aligned on column names to avoid silent mis-filtering.
        if hasattr(self.strategy, "date_col"):
            self.strategy.date_col = self.date_col
        if hasattr(self.strategy, "ticker_col"):
            self.strategy.ticker_col = self.ticker_col
        if hasattr(self.strategy, "price_col"):
            self.strategy.price_col = self.price_col

    def _normalize_contribution_schedule(self, contribution_schedule: Optional[Dict]) -> Dict[pd.Timestamp, float]:
        out: Dict[pd.Timestamp, float] = {}
        if not contribution_schedule:
            return out
        for k, v in contribution_schedule.items():
            ts = pd.to_datetime(k, errors="coerce")
            if pd.isna(ts):
                raise ValueError(f"Invalid contribution_schedule date key: {k}")
            out[pd.Timestamp(ts).normalize()] = float(v)
        return out

    def _contribution_for_date(self, date: pd.Timestamp) -> float:
        d = pd.Timestamp(date).normalize()
        return self.periodic_contribution + self.contribution_schedule.get(d, 0.0)

    def _get_rebalance_dates(self) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency."""
        if self.rebalance_frequency == 'monthly':
            # Group by month, take last trading day of each month
            year_month = self.data[self.date_col].dt.to_period('M')
            dates = self.data.groupby(year_month)[self.date_col].max().tolist()
            return sorted(dates)
        elif self.rebalance_frequency == 'daily':
            return sorted(self.data[self.date_col].unique())
        else:
            raise ValueError(f"Invalid rebalance_frequency: {self.rebalance_frequency}")

    def run(self) -> BacktestResult:
        """Execute backtest and return results."""
        rebalance_dates = self._get_rebalance_dates()

        capital = self.initial_capital
        portfolio_values = []
        returns_list = []
        turnover_list = []
        cash_flow_list = []
        positions_list = []

        prev_shares = {}  # ticker -> shares held

        for i, date in enumerate(rebalance_dates):
            cash_flow = self._contribution_for_date(date)
            if cash_flow != 0:
                capital += cash_flow
                if capital <= 0:
                    raise ValueError(
                        f"Capital became non-positive after cash flow on {date}: {capital:.2f}. "
                        f"Adjust contributions/withdrawals."
                    )
            cash_flow_list.append(cash_flow)

            # Get data for this period
            period_data = self.data[self.data[self.date_col] == date]
            period_data = self._collapse_period_data(period_data)

            if period_data.empty:
                portfolio_values.append(capital)
                returns_list.append(0.0)
                turnover_list.append(0.0)
                continue

            # Get strategy signals (weights)
            target_weights = self.strategy.select(self.data, date) or {}
            target_weights = {
                str(t): float(w)
                for t, w in target_weights.items()
                if pd.notna(w) and np.isfinite(w)
            }

            if not target_weights:
                # Cash position - liquidate everything
                shares = {}
            else:
                # Convert weights to shares
                shares = self._weights_to_shares(
                    target_weights,
                    capital,
                    period_data
                )

            turnover = self._calculate_turnover(prev_shares, shares, period_data, capital)
            turnover_list.append(turnover)

            # Record positions
            for ticker, qty in shares.items():
                if ticker in period_data[self.ticker_col].values:
                    price = period_data[period_data[self.ticker_col] == ticker][self.price_col].values[0]
                    weight = target_weights.get(ticker, 0)
                    positions_list.append({
                        'date': date,
                        'ticker': ticker,
                        'shares': qty,
                        'price': price,
                        'weight': weight,
                        'position_value': qty * price
                    })

            # Calculate period return
            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
                period_return = self._calculate_period_return(
                    shares,
                    date,
                    next_date,
                    turnover,
                    capital
                )

                # Apply taxes on gains
                gross_profit = capital * period_return
                if gross_profit > 0:
                    tax = gross_profit * self.short_term_tax_rate
                    period_return = (gross_profit - tax) / capital

                capital = capital * (1 + period_return)
                returns_list.append(period_return)
            else:
                # Last period - no return to calculate
                returns_list.append(0.0)

            portfolio_values.append(capital)
            prev_shares = shares

        # Create result series
        equity_curve = pd.Series(portfolio_values, index=rebalance_dates)
        returns_series = pd.Series(returns_list, index=rebalance_dates)
        turnover_series = pd.Series(turnover_list, index=rebalance_dates)
        cash_flows_series = pd.Series(cash_flow_list, index=rebalance_dates)
        positions_df = pd.DataFrame(positions_list)

        # Calculate metrics
        periods_per_year = 12 if self.rebalance_frequency == 'monthly' else 252
        metrics = compute_metrics(
            returns_series,
            benchmark_returns=None,  # Will add benchmark comparison below
            turnover=turnover_series,
            periods_per_year=periods_per_year
        )

        # Benchmark comparison
        benchmark_equity, benchmark_returns, benchmark_metrics = self._calculate_benchmark(
            rebalance_dates,
            periods_per_year
        )

        # Re-compute metrics with benchmark
        benchmark_for_metrics = benchmark_returns if benchmark_metrics else None
        metrics = compute_metrics(
            returns_series,
            benchmark_returns=benchmark_for_metrics,
            turnover=turnover_series,
            periods_per_year=periods_per_year
        )

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns_series,
            positions=positions_df,
            turnover=turnover_series,
            metrics=metrics,
            benchmark_equity=benchmark_equity,
            benchmark_returns=benchmark_returns,
            benchmark_metrics=benchmark_metrics,
            cash_flows=cash_flows_series,
        )

    def _collapse_period_data(self, period_data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure one row per ticker for a rebalance bucket.

        When date_col is coarse (e.g., month_end) and input is daily, many rows
        can share the same (ticker, date_col). We keep the latest available row.
        """
        if period_data.empty:
            return period_data
        if not period_data[self.ticker_col].duplicated().any():
            return period_data

        sort_cols: List[str] = []
        if self.date_col != "date" and "date" in period_data.columns:
            sort_cols.append("date")
        if self.date_col != "timestamp" and "timestamp" in period_data.columns:
            sort_cols.append("timestamp")
        if not sort_cols:
            sort_cols = [self.date_col]

        collapsed = (
            period_data.sort_values(sort_cols)
            .groupby(self.ticker_col, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        return collapsed

    def _weights_to_shares(
        self,
        weights: Dict[str, float],
        capital: float,
        period_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Convert target weights to shares."""
        shares = {}

        for ticker, weight in weights.items():
            # Get price for this ticker
            ticker_data = period_data[period_data[self.ticker_col] == ticker]
            if ticker_data.empty:
                continue

            price = ticker_data[self.price_col].values[0]
            if price <= 0:
                continue

            # Calculate shares
            # weight > 0 = long, weight < 0 = short
            alloc = capital * self.buying_power_multiplier * weight
            shares[ticker] = alloc / price

        return shares

    def _calculate_turnover(
        self,
        prev_shares: Dict[str, float],
        new_shares: Dict[str, float],
        period_data: pd.DataFrame,
        capital: float
    ) -> float:
        """
        Calculate portfolio turnover as:
        min(total purchases, total sales) / average portfolio value.
        """
        all_tickers = set(prev_shares.keys()) | set(new_shares.keys())

        purchases = 0.0
        sales = 0.0
        prev_gross = 0.0
        new_gross = 0.0

        for ticker in all_tickers:
            # Get prices
            ticker_data = period_data[period_data[self.ticker_col] == ticker]
            if ticker_data.empty:
                continue
            price = ticker_data[self.price_col].values[0]

            # Calculate value change
            prev_val = prev_shares.get(ticker, 0) * price
            new_val = new_shares.get(ticker, 0) * price

            prev_gross += abs(prev_val)
            new_gross += abs(new_val)

            delta = new_val - prev_val
            if delta > 0:
                purchases += delta
            elif delta < 0:
                sales += -delta

        avg_portfolio_value = 0.5 * (prev_gross + new_gross)
        if avg_portfolio_value <= 0:
            avg_portfolio_value = capital

        if avg_portfolio_value <= 0:
            return 0.0

        return min(purchases, sales) / avg_portfolio_value

    def _calculate_period_return(
        self,
        shares: Dict[str, float],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        turnover: float,
        capital: float
    ) -> float:
        """Calculate portfolio return for a period including costs."""
        start_data = self._collapse_period_data(self.data[self.data[self.date_col] == start_date])
        end_data = self._collapse_period_data(self.data[self.data[self.date_col] == end_date])

        pnl = 0.0
        borrow_costs = 0.0

        for ticker, qty in shares.items():
            # Start price
            start_ticker = start_data[start_data[self.ticker_col] == ticker]
            if start_ticker.empty:
                continue
            start_price = start_ticker[self.price_col].values[0]

            # End price
            end_ticker = end_data[end_data[self.ticker_col] == ticker]
            if end_ticker.empty:
                # Missing end price is often a panel-construction artifact
                # (e.g., prediction merge drops a ticker for that date bucket).
                # Default is to carry forward start price to avoid synthetic crashes.
                if self.missing_price_mode == "carry":
                    end_price = start_price
                else:
                    if qty > 0:
                        end_price = 0.0
                    else:
                        # Conservative short fallback: +100% move in underlying.
                        end_price = start_price * 2.0
            else:
                end_price = end_ticker[self.price_col].values[0]

            raw_ret = _safe_price_return(start_price, end_price)
            if self.max_abs_ticker_return is not None and pd.notna(raw_ret):
                raw_ret = float(np.clip(raw_ret, -self.max_abs_ticker_return, self.max_abs_ticker_return))

            # Works for both long and short using clipped close-to-close return.
            notional = qty * start_price
            pnl += notional * raw_ret

            if qty < 0:
                period_fraction = 1 / 12 if self.rebalance_frequency == 'monthly' else 1 / 252
                borrow_costs += abs(qty * start_price) * self.borrow_cost_annual * period_fraction

        # Transaction costs
        transaction_costs = turnover * capital * self.buying_power_multiplier * (self.transaction_cost_bps / 10000)

        # Calculate return
        if capital > 0:
            period_return = (pnl - transaction_costs - borrow_costs) / capital
        else:
            period_return = 0.0

        # Prevent one pathological period from forcing impossible negative equity.
        period_return = max(period_return, self.liquidation_floor)
        return period_return

    def _calculate_benchmark(
        self,
        rebalance_dates: List[pd.Timestamp],
        periods_per_year: int
    ) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
        """Calculate benchmark equity curve and metrics."""
        # Filter benchmark data
        benchmark_data = self.data[self.data[self.ticker_col] == self.benchmark_ticker].copy()

        if benchmark_data.empty:
            # No benchmark data - return empty results
            empty_series = pd.Series(index=rebalance_dates, data=0.0)
            return empty_series, empty_series, {}

        # Collapse duplicate rows within each rebalance bucket (e.g., daily rows sharing month_end)
        # For a single benchmark ticker we need one row per date bucket, not per ticker.
        if not benchmark_data.empty and benchmark_data[self.date_col].duplicated().any():
            sort_cols: List[str] = []
            if self.date_col != "date" and "date" in benchmark_data.columns:
                sort_cols.append("date")
            if self.date_col != "timestamp" and "timestamp" in benchmark_data.columns:
                sort_cols.append("timestamp")
            if not sort_cols:
                sort_cols = [self.date_col]

            benchmark_data = (
                benchmark_data.sort_values(sort_cols)
                .groupby(self.date_col, as_index=False)
                .tail(1)
                .reset_index(drop=True)
            )

        # Get prices at rebalance dates
        benchmark_prices = []
        for date in rebalance_dates:
            date_data = benchmark_data[benchmark_data[self.date_col] == date]
            if not date_data.empty:
                benchmark_prices.append(date_data[self.price_col].values[0])
            else:
                # Use last known price
                prior_data = benchmark_data[benchmark_data[self.date_col] <= date]
                if not prior_data.empty:
                    benchmark_prices.append(prior_data[self.price_col].values[-1])
                else:
                    benchmark_prices.append(np.nan)

        # Calculate returns
        benchmark_prices_series = pd.Series(benchmark_prices, index=rebalance_dates)
        benchmark_returns = benchmark_prices_series.pct_change().fillna(0)

        # Calculate equity curve
        benchmark_equity = self.initial_capital * (1 + benchmark_returns).cumprod()

        # Calculate metrics
        benchmark_metrics = compute_metrics(
            benchmark_returns,
            benchmark_returns=None,
            turnover=None,
            periods_per_year=periods_per_year
        )

        return benchmark_equity, benchmark_returns, benchmark_metrics
