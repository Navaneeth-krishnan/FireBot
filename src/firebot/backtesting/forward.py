"""Forward testing mode for simulated live trading.

Forward testing extends the backtest engine to support streaming
bar-by-bar execution, mimicking live market conditions. Unlike
backtesting which runs over a fixed dataset, forward testing
allows pausing, inspecting state, and resuming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from firebot.core.models import OHLCV, Order, OrderSide, OrderType, SignalDirection
from firebot.execution.engine import PaperTradingEngine
from firebot.execution.portfolio import PortfolioSimulator
from firebot.strategies.base import Strategy


@dataclass
class ForwardTestState:
    """Snapshot of forward test state at a point in time."""

    bar_count: int
    portfolio_value: Decimal
    cash: Decimal
    num_positions: int
    num_trades: int
    is_paused: bool
    equity_curve: list[Decimal]


class ForwardTestRunner:
    """Forward testing runner for simulated live execution.

    Processes bars one at a time, maintaining full state between
    calls. Supports pause/resume and state inspection.

    Example:
        runner = ForwardTestRunner(
            strategy=my_strategy,
            initial_capital=Decimal("100000"),
            position_size_pct=Decimal("0.1"),
        )
        for bar in data_stream:
            runner.on_bar(bar)
            state = runner.get_state()
            if state.portfolio_value < Decimal("90000"):
                runner.pause()
                break
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: Decimal,
        position_size_pct: Decimal = Decimal("0.1"),
        slippage_bps: int = 0,
        commission_per_trade: float = 0,
    ) -> None:
        self.strategy = strategy
        self._trading_engine = PaperTradingEngine(
            fill_model="instant",
            slippage_bps=slippage_bps,
            commission_per_trade=commission_per_trade,
        )
        self._portfolio = PortfolioSimulator(
            strategy_id=strategy.strategy_id,
            initial_capital=initial_capital,
        )
        self._position_size_pct = position_size_pct
        self._bar_count = 0
        self._total_trades = 0
        self._equity_curve: list[Decimal] = []
        self._paused = False

    @property
    def is_paused(self) -> bool:
        """Check if forward test is paused."""
        return self._paused

    @property
    def bar_count(self) -> int:
        """Number of bars processed."""
        return self._bar_count

    def pause(self) -> None:
        """Pause forward testing."""
        self._paused = True

    def resume(self) -> None:
        """Resume forward testing."""
        self._paused = False

    def on_bar(self, bar: OHLCV) -> bool:
        """Process a single bar.

        Args:
            bar: New OHLCV bar

        Returns:
            True if bar was processed, False if paused
        """
        if self._paused:
            return False

        self._bar_count += 1

        # Feed to strategy
        self.strategy.on_data(bar)

        # Update prices
        self._portfolio.update_price(bar.symbol, bar.close)

        # Check pending orders
        current_prices = {bar.symbol: bar.close}
        pending_fills = self._trading_engine.check_pending_orders(current_prices)
        for fill_result in pending_fills:
            if fill_result.status == "filled" and fill_result.fill_price is not None:
                order_side = self._find_order_side(fill_result.order_id)
                if order_side is not None and fill_result.fill_quantity is not None:
                    self._portfolio.execute_fill(
                        symbol=bar.symbol,
                        side=order_side,
                        quantity=fill_result.fill_quantity,
                        price=fill_result.fill_price,
                    )
                    self._total_trades += 1

        # Generate signal
        features = {
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "returns": 0.0,
        }
        signal = self.strategy.generate_signal(features)

        # Execute if actionable
        if signal is not None and signal.direction != SignalDirection.NEUTRAL:
            side = (
                OrderSide.BUY
                if signal.direction == SignalDirection.LONG
                else OrderSide.SELL
            )

            if self._portfolio.cash > 0 and bar.close > 0:
                allocation = self._portfolio.cash * self._position_size_pct
                quantity = Decimal(str(int(allocation / bar.close)))

                if quantity > 0:
                    order = Order(
                        id=f"fwd_{self._bar_count}",
                        timestamp=bar.timestamp,
                        symbol=bar.symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        price=bar.close,
                        strategy_id=self.strategy.strategy_id,
                    )

                    fill_result = self._trading_engine.submit_order(order, bar.close)

                    if fill_result.status == "filled" and fill_result.fill_price is not None:
                        self._portfolio.execute_fill(
                            symbol=bar.symbol,
                            side=side,
                            quantity=fill_result.fill_quantity or quantity,
                            price=fill_result.fill_price,
                        )
                        self._total_trades += 1

        self._equity_curve.append(self._portfolio.total_value)
        return True

    def get_state(self) -> ForwardTestState:
        """Get current state snapshot.

        Returns:
            ForwardTestState with current metrics
        """
        return ForwardTestState(
            bar_count=self._bar_count,
            portfolio_value=self._portfolio.total_value,
            cash=self._portfolio.cash,
            num_positions=len(self._portfolio.positions),
            num_trades=self._total_trades,
            is_paused=self._paused,
            equity_curve=list(self._equity_curve),
        )

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary.

        Returns:
            Portfolio summary dict
        """
        return self._portfolio.get_summary()

    def _find_order_side(self, order_id: str) -> OrderSide | None:
        """Find the side of an order from the trading engine history."""
        for order, _ in self._trading_engine.order_history:
            if order.id == order_id:
                return order.side
        return None
