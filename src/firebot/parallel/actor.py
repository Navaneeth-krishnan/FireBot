"""Strategy Actor for parallel execution."""

from decimal import Decimal
from typing import Any

from firebot.core.models import OHLCV, Signal
from firebot.execution.portfolio import PortfolioSimulator
from firebot.strategies.base import Strategy


class StrategyActor:
    """Actor wrapper for strategy execution with isolated portfolio.

    Each StrategyActor encapsulates:
    - A strategy instance
    - An independent portfolio simulator
    - Data buffer for features

    This enables parallel execution where each strategy has its own
    isolated state and portfolio.

    Example:
        actor = StrategyActor(
            strategy_id="momentum_1",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 20},
            initial_capital=Decimal("100000"),
        )
        actor.on_data(ohlcv)
        state = actor.get_state()
    """

    def __init__(
        self,
        strategy_id: str,
        strategy_class: type[Strategy],
        config: dict[str, Any],
        initial_capital: Decimal,
        max_drawdown_pct: Decimal = Decimal("0.10"),
        max_position_size_pct: Decimal = Decimal("0.05"),
    ) -> None:
        """Initialize strategy actor.

        Args:
            strategy_id: Unique identifier for this strategy instance
            strategy_class: Strategy class to instantiate
            config: Configuration dict for strategy
            initial_capital: Starting capital for portfolio
            max_drawdown_pct: Maximum drawdown before strategy is disabled
            max_position_size_pct: Maximum position size as % of portfolio
        """
        self.strategy_id = strategy_id
        self.config = config

        # Create strategy instance
        self.strategy = strategy_class(strategy_id=strategy_id, config=config)

        # Create independent portfolio
        self.portfolio = PortfolioSimulator(
            strategy_id=strategy_id,
            initial_capital=initial_capital,
            max_drawdown_pct=max_drawdown_pct,
            max_position_size_pct=max_position_size_pct,
        )

        # Data buffer for features
        self._data_buffer: list[OHLCV] = []
        self._is_disabled = False

    def on_data(self, ohlcv: OHLCV) -> Signal | None:
        """Process new market data.

        Args:
            ohlcv: New market data point

        Returns:
            Signal if strategy generates one, None otherwise
        """
        if self._is_disabled:
            return None

        # Check if drawdown breached
        if self.portfolio.is_drawdown_breached():
            self._is_disabled = True
            return None

        # Add to buffer and pass to strategy
        self._data_buffer.append(ohlcv)
        self.strategy.on_data(ohlcv)

        # Update price in portfolio
        self.portfolio.update_price(ohlcv.symbol, ohlcv.close)

        return None  # Signal generation would happen in a full implementation

    def get_state(self) -> dict[str, Any]:
        """Get current actor state.

        Returns:
            Dict containing strategy and portfolio state
        """
        return {
            "strategy_id": self.strategy_id,
            "config": self.config,
            "is_disabled": self._is_disabled,
            "portfolio": {
                "cash": self.portfolio.cash,
                "positions": {k: v.model_dump() for k, v in self.portfolio.positions.items()},
                "total_value": self.portfolio.total_value,
                "realized_pnl": self.portfolio.realized_pnl,
                "drawdown": self.portfolio.drawdown,
                "high_water_mark": self.portfolio.high_water_mark,
            },
            "data_buffer_size": len(self._data_buffer),
        }

    def reset(self) -> None:
        """Reset actor state to initial conditions."""
        self._data_buffer = []
        self._is_disabled = False
        self.portfolio = PortfolioSimulator(
            strategy_id=self.strategy_id,
            initial_capital=self.portfolio.high_water_mark or Decimal("100000"),
            max_drawdown_pct=self.portfolio._max_drawdown_pct,
            max_position_size_pct=self.portfolio._max_position_size_pct,
        )
