"""Tests for forward testing mode."""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any

import pytest

from firebot.core.models import OHLCV, Signal, SignalDirection
from firebot.backtesting.forward import ForwardTestRunner, ForwardTestState
from firebot.strategies.base import Strategy


class AlwaysLongStrategy(Strategy):
    """Strategy that always goes LONG."""

    def on_data(self, data: OHLCV) -> None:
        self.data_buffer.append(data)

    def generate_signal(self, features: dict[str, Any]) -> Signal | None:
        if not self.data_buffer:
            return None
        return Signal(
            timestamp=self.data_buffer[-1].timestamp,
            symbol=self.data_buffer[-1].symbol,
            direction=SignalDirection.LONG,
            confidence=0.8,
            strategy_id=self.strategy_id,
        )


def _make_bar(i: int, price: float = 150.0) -> OHLCV:
    """Create a test OHLCV bar."""
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
    p = Decimal(str(round(price + i * 0.5, 2)))
    return OHLCV(
        timestamp=ts,
        symbol="AAPL",
        open=p - Decimal("0.5"),
        high=p + Decimal("1"),
        low=p - Decimal("1"),
        close=p,
        volume=Decimal("1000000"),
    )


class TestForwardTestRunner:
    """Tests for forward test runner."""

    def test_initialization(self) -> None:
        """Runner should initialize with strategy and capital."""
        strategy = AlwaysLongStrategy("test_1", {})
        runner = ForwardTestRunner(
            strategy=strategy,
            initial_capital=Decimal("100000"),
        )
        assert runner.bar_count == 0
        assert not runner.is_paused

    def test_process_bar(self) -> None:
        """Should process a single bar and update state."""
        strategy = AlwaysLongStrategy("test_1", {})
        runner = ForwardTestRunner(
            strategy=strategy,
            initial_capital=Decimal("100000"),
        )
        bar = _make_bar(0)
        result = runner.on_bar(bar)
        assert result is True
        assert runner.bar_count == 1

    def test_multiple_bars(self) -> None:
        """Should process multiple bars sequentially."""
        strategy = AlwaysLongStrategy("test_1", {})
        runner = ForwardTestRunner(
            strategy=strategy,
            initial_capital=Decimal("100000"),
        )
        for i in range(10):
            runner.on_bar(_make_bar(i))
        assert runner.bar_count == 10

    def test_pause_resume(self) -> None:
        """Paused runner should not process bars."""
        strategy = AlwaysLongStrategy("test_1", {})
        runner = ForwardTestRunner(
            strategy=strategy,
            initial_capital=Decimal("100000"),
        )
        runner.on_bar(_make_bar(0))
        assert runner.bar_count == 1

        runner.pause()
        assert runner.is_paused
        result = runner.on_bar(_make_bar(1))
        assert result is False
        assert runner.bar_count == 1  # Unchanged

        runner.resume()
        assert not runner.is_paused
        runner.on_bar(_make_bar(2))
        assert runner.bar_count == 2

    def test_get_state(self) -> None:
        """Should return accurate state snapshot."""
        strategy = AlwaysLongStrategy("test_1", {})
        runner = ForwardTestRunner(
            strategy=strategy,
            initial_capital=Decimal("100000"),
        )
        for i in range(5):
            runner.on_bar(_make_bar(i))

        state = runner.get_state()
        assert isinstance(state, ForwardTestState)
        assert state.bar_count == 5
        assert state.portfolio_value > 0
        assert len(state.equity_curve) == 5

    def test_equity_curve_grows(self) -> None:
        """Equity curve should have one entry per bar."""
        strategy = AlwaysLongStrategy("test_1", {})
        runner = ForwardTestRunner(
            strategy=strategy,
            initial_capital=Decimal("100000"),
        )
        for i in range(8):
            runner.on_bar(_make_bar(i))

        state = runner.get_state()
        assert len(state.equity_curve) == 8

    def test_portfolio_summary(self) -> None:
        """Should return portfolio summary dict."""
        strategy = AlwaysLongStrategy("test_1", {})
        runner = ForwardTestRunner(
            strategy=strategy,
            initial_capital=Decimal("100000"),
        )
        runner.on_bar(_make_bar(0))

        summary = runner.get_portfolio_summary()
        assert "cash" in summary
        assert "total_value" in summary
        assert "strategy_id" in summary
