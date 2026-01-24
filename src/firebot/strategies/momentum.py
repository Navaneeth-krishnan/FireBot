"""Momentum-based trading strategy implementation."""

from datetime import datetime, timezone
from typing import Any

from firebot.core.models import OHLCV, Signal, SignalDirection
from firebot.strategies.base import Strategy


class MomentumStrategy(Strategy):
    """Simple momentum-based trading strategy.

    Generates signals based on price momentum over a lookback window.
    - LONG when momentum exceeds positive threshold
    - SHORT when momentum falls below negative threshold
    - NEUTRAL when within threshold bounds

    Configuration:
        lookback_window: Number of bars to consider (default: 20)
        threshold: Minimum momentum to generate signal (default: 0.02 = 2%)
        max_buffer_size: Maximum data buffer size (default: 1000)
    """

    def __init__(self, strategy_id: str, config: dict[str, Any]) -> None:
        """Initialize momentum strategy.

        Args:
            strategy_id: Unique identifier
            config: Configuration with optional keys:
                - lookback_window: int
                - threshold: float
                - max_buffer_size: int
        """
        super().__init__(strategy_id, config)
        self.lookback_window: int = config.get("lookback_window", 20)
        self.threshold: float = config.get("threshold", 0.02)
        self.max_buffer_size: int = config.get("max_buffer_size", 1000)
        self._last_symbol: str = ""

    def on_data(self, data: OHLCV) -> None:
        """Process new market data.

        Adds data to buffer and maintains max buffer size.

        Args:
            data: New OHLCV bar
        """
        self.data_buffer.append(data)
        self._last_symbol = data.symbol

        # Trim buffer if too large
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size :]

    def generate_signal(self, features: dict[str, Any]) -> Signal | None:
        """Generate trading signal based on momentum.

        Args:
            features: Must contain 'returns' key with momentum value

        Returns:
            Signal with direction based on momentum, or None if no returns data
        """
        returns = features.get("returns")
        if returns is None:
            return None

        # Determine direction based on threshold
        if returns > self.threshold:
            direction = SignalDirection.LONG
        elif returns < -self.threshold:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        # Calculate confidence based on momentum strength
        # Normalize to 0-1 range, cap at 1.0
        momentum_strength = abs(returns)
        confidence = min(momentum_strength / (self.threshold * 5), 1.0)

        # Get timestamp from last data or use current time
        if self.data_buffer:
            timestamp = self.data_buffer[-1].timestamp
            symbol = self.data_buffer[-1].symbol
        else:
            timestamp = datetime.now(timezone.utc)
            symbol = self._last_symbol or "UNKNOWN"

        return Signal(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            strategy_id=self.strategy_id,
            metadata={
                "returns": returns,
                "threshold": self.threshold,
                "lookback_window": self.lookback_window,
            },
        )

    def calculate_momentum(self) -> float | None:
        """Calculate momentum from data buffer.

        Returns:
            Momentum as percentage change, or None if insufficient data
        """
        if len(self.data_buffer) < self.lookback_window:
            return None

        recent = self.data_buffer[-self.lookback_window :]
        start_price = float(recent[0].close)
        end_price = float(recent[-1].close)

        if start_price == 0:
            return None

        return (end_price - start_price) / start_price
