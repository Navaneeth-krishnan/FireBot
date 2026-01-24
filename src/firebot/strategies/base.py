"""Base strategy interface for FireBot trading strategies."""

from abc import ABC, abstractmethod
from typing import Any

from firebot.core.models import OHLCV, Signal


class Strategy(ABC):
    """Abstract base class for trading strategies.

    All strategies must inherit from this class and implement:
    - on_data(): Called when new market data arrives
    - generate_signal(): Generate trading signal from features

    Example:
        class MyStrategy(Strategy):
            def on_data(self, data: OHLCV) -> None:
                self.data_buffer.append(data)

            def generate_signal(self, features: dict) -> Signal | None:
                if features["sma_20"] > features["sma_50"]:
                    return Signal(...)
                return None
    """

    def __init__(self, strategy_id: str, config: dict[str, Any]) -> None:
        """Initialize strategy.

        Args:
            strategy_id: Unique identifier for this strategy instance
            config: Configuration parameters for the strategy
        """
        self.strategy_id = strategy_id
        self.config = config
        self.data_buffer: list[OHLCV] = []

    @abstractmethod
    def on_data(self, data: OHLCV) -> None:
        """Called when new market data arrives.

        Strategies should use this to update internal state, buffers,
        or trigger calculations.

        Args:
            data: New OHLCV bar
        """
        pass

    @abstractmethod
    def generate_signal(self, features: dict[str, Any]) -> Signal | None:
        """Generate a trading signal from computed features.

        Args:
            features: Dictionary of computed features from FeaturePipeline

        Returns:
            Signal if a trade should be made, None otherwise
        """
        pass

    def on_fill(self, order: Any) -> None:
        """Called when an order is filled.

        Override this method to handle order fill notifications.

        Args:
            order: The filled order
        """
        pass

    def reset(self) -> None:
        """Reset strategy state.

        Call this between backtests or when restarting.
        """
        self.data_buffer.clear()
