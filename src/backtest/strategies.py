"""
Trading strategy classes.

Defines selection logic for backtesting.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional


class Strategy(ABC):
    """Base class for trading strategies."""

    @abstractmethod
    def select(self, data: pd.DataFrame, date: pd.Timestamp) -> Dict[str, float]:
        """
        Select stocks and return weights for given date.

        Args:
            data: Full dataset (pre-filtered to available data up to date)
            date: Current rebalance date

        Returns:
            Dict mapping ticker -> weight
            - Positive weights = long positions
            - Negative weights = short positions
            - Weights should sum to 1.0 for long-only, or 0.0 for dollar-neutral
        """
        pass


class LongOnlyTopPct(Strategy):
    """
    Long-only strategy: Buy top N% by prediction.

    Equal-weights the top percentile of stocks based on prediction scores.
    """

    def __init__(
        self,
        pred_col: str = 'pred',
        top_pct: float = 0.20,
        date_col: str = 'date',
        ticker_col: str = 'ticker',
        price_col: str = 'close',
        min_price: Optional[float] = 1.0,
    ):
        """
        Args:
            pred_col: Column name containing predictions
            top_pct: Top percentile to select (0.20 = top 20%)
        """
        self.pred_col = pred_col
        self.top_pct = top_pct
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.price_col = price_col
        self.min_price = min_price

    def select(self, data: pd.DataFrame, date: pd.Timestamp) -> Dict[str, float]:
        """Select top top_pct% of stocks, equal-weighted."""
        # Filter to this date
        day_data = data[data[self.date_col] == date].copy()

        if day_data.empty or self.pred_col not in day_data.columns:
            return {}

        # Remove missing predictions
        day_data = day_data.dropna(subset=[self.pred_col])
        day_data = self._dedupe_cross_section(day_data)
        if self.min_price is not None and self.price_col in day_data.columns:
            day_data = day_data[pd.to_numeric(day_data[self.price_col], errors="coerce") >= self.min_price]

        if len(day_data) == 0:
            return {}

        # Get cutoff for top percentile
        cutoff = day_data[self.pred_col].quantile(1 - self.top_pct)
        selected = day_data[day_data[self.pred_col] >= cutoff]

        if len(selected) == 0:
            return {}

        # Equal-weight
        weight = 1.0 / len(selected)
        return {ticker: weight for ticker in selected[self.ticker_col]}

    def _dedupe_cross_section(self, day_data: pd.DataFrame) -> pd.DataFrame:
        """Keep one row per ticker for this rebalance bucket."""
        if day_data.empty or not day_data[self.ticker_col].duplicated().any():
            return day_data
        sort_cols = []
        if self.date_col != "date" and "date" in day_data.columns:
            sort_cols.append("date")
        if self.date_col != "timestamp" and "timestamp" in day_data.columns:
            sort_cols.append("timestamp")
        if not sort_cols:
            sort_cols = [self.date_col]
        return (
            day_data.sort_values(sort_cols)
            .groupby(self.ticker_col, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )


class LongShortTopBottom(Strategy):
    """
    Long/short strategy: Long top N%, short bottom M%.

    Supports dollar-neutral (50% long, 50% short) or asymmetric weighting.
    For non-dollar-neutral mode, exposures are scaled so gross exposure is 1.0.
    """

    def __init__(
        self,
        pred_col: str = 'pred',
        long_pct: float = 0.20,
        short_pct: float = 0.20,
        dollar_neutral: bool = True,
        date_col: str = 'date',
        ticker_col: str = 'ticker',
        price_col: str = 'close',
        min_price: Optional[float] = 1.0,
    ):
        """
        Args:
            pred_col: Column name containing predictions
            long_pct: Top percentile to long (0.20 = top 20%)
            short_pct: Bottom percentile to short (0.20 = bottom 20%)
            dollar_neutral: If True, 50% long exposure + 50% short exposure
        """
        self.pred_col = pred_col
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.dollar_neutral = dollar_neutral
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.price_col = price_col
        self.min_price = min_price

    def select(self, data: pd.DataFrame, date: pd.Timestamp) -> Dict[str, float]:
        """Select top/bottom percentiles with specified weighting."""
        # Filter to this date
        day_data = data[data[self.date_col] == date].copy()

        if day_data.empty or self.pred_col not in day_data.columns:
            return {}

        # Remove missing predictions
        day_data = day_data.dropna(subset=[self.pred_col])
        day_data = self._dedupe_cross_section(day_data)
        if self.min_price is not None and self.price_col in day_data.columns:
            day_data = day_data[pd.to_numeric(day_data[self.price_col], errors="coerce") >= self.min_price]

        if len(day_data) == 0:
            return {}

        weights = {}

        # Long selection (top percentile)
        if self.long_pct > 0:
            long_cutoff = day_data[self.pred_col].quantile(1 - self.long_pct)
            long_selected = day_data[day_data[self.pred_col] >= long_cutoff]

            if len(long_selected) > 0:
                if self.dollar_neutral and self.short_pct > 0:
                    long_exposure = 0.5  # 50% long in dollar-neutral
                else:
                    if self.short_pct > 0:
                        total_pct = self.long_pct + self.short_pct
                        long_exposure = self.long_pct / total_pct if total_pct > 0 else 1.0
                    else:
                        long_exposure = 1.0

                long_weight = long_exposure / len(long_selected)
                for ticker in long_selected[self.ticker_col]:
                    weights[ticker] = long_weight

        # Short selection (bottom percentile)
        if self.short_pct > 0:
            short_cutoff = day_data[self.pred_col].quantile(self.short_pct)
            short_selected = day_data[day_data[self.pred_col] <= short_cutoff]
            if self.long_pct > 0 and len(weights) > 0:
                long_tickers = set(weights.keys())
                short_selected = short_selected[~short_selected[self.ticker_col].isin(long_tickers)]

            if len(short_selected) > 0:
                if self.dollar_neutral:
                    short_exposure = -0.5  # -50% short in dollar-neutral
                else:
                    if self.long_pct > 0:
                        total_pct = self.long_pct + self.short_pct
                        short_exposure = -(self.short_pct / total_pct) if total_pct > 0 else 0.0
                    else:
                        short_exposure = -1.0

                short_weight = short_exposure / len(short_selected)
                for ticker in short_selected[self.ticker_col]:
                    weights[ticker] = short_weight

        return weights

    def _dedupe_cross_section(self, day_data: pd.DataFrame) -> pd.DataFrame:
        """Keep one row per ticker for this rebalance bucket."""
        if day_data.empty or not day_data[self.ticker_col].duplicated().any():
            return day_data
        sort_cols = []
        if self.date_col != "date" and "date" in day_data.columns:
            sort_cols.append("date")
        if self.date_col != "timestamp" and "timestamp" in day_data.columns:
            sort_cols.append("timestamp")
        if not sort_cols:
            sort_cols = [self.date_col]
        return (
            day_data.sort_values(sort_cols)
            .groupby(self.ticker_col, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )


class RegimeConditionalStrategy(Strategy):
    """
    Deploy sub-strategies based on regime predictions.

    Uses a regime prediction column to switch between different strategies
    (e.g., HMM states, market regime classifier output).
    """

    def __init__(
        self,
        regime_col: str = 'regime_pred',
        regime_strategies: Optional[Dict[int, Strategy]] = None,
        date_col: str = 'date',
    ):
        """
        Args:
            regime_col: Column with regime predictions (integer labels)
            regime_strategies: Dict mapping regime ID -> Strategy
                Example: {
                    0: CashStrategy(),           # Bear market
                    1: LongOnlyTopPct(),          # Bull market
                    2: LongShortTopBottom()       # Sideways market
                }
        """
        self.regime_col = regime_col
        self.regime_strategies = regime_strategies or {}
        self.date_col = date_col

    def select(self, data: pd.DataFrame, date: pd.Timestamp) -> Dict[str, float]:
        """Dispatch to sub-strategy based on current regime."""
        # Get current regime
        day_data = data[data[self.date_col] == date]

        if day_data.empty or self.regime_col not in day_data.columns:
            return {}

        # Assume regime is constant across all tickers for a given date
        # (typically a market-level prediction)
        regime_values = day_data[self.regime_col].dropna()

        if len(regime_values) == 0:
            return {}

        current_regime = int(regime_values.iloc[0])

        # Get strategy for this regime
        strategy = self.regime_strategies.get(current_regime)

        if strategy is None:
            # No strategy defined for this regime - hold cash
            return {}

        # Delegate to sub-strategy
        return strategy.select(data, date)


class CashStrategy(Strategy):
    """
    Hold cash (no positions).

    Useful for regime-based strategies during unfavorable market conditions.
    """

    def select(self, data: pd.DataFrame, date: pd.Timestamp) -> Dict[str, float]:
        """Return empty positions (100% cash)."""
        return {}
