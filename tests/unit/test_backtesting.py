"""Tests for backtesting framework."""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any

import pytest

from firebot.core.models import OHLCV, Signal, SignalDirection
from firebot.backtesting.engine import BacktestEngine, BacktestConfig, BacktestResult
from firebot.strategies.base import Strategy


class SimpleTestStrategy(Strategy):
    """Strategy that always goes LONG for testing."""

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


class AlternatingStrategy(Strategy):
    """Strategy that alternates LONG/SHORT for testing."""

    def __init__(self, strategy_id: str, config: dict[str, Any]) -> None:
        super().__init__(strategy_id, config)
        self._step = 0

    def on_data(self, data: OHLCV) -> None:
        self.data_buffer.append(data)
        self._step += 1

    def generate_signal(self, features: dict[str, Any]) -> Signal | None:
        if not self.data_buffer:
            return None
        direction = SignalDirection.LONG if self._step % 2 == 0 else SignalDirection.SHORT
        return Signal(
            timestamp=self.data_buffer[-1].timestamp,
            symbol=self.data_buffer[-1].symbol,
            direction=direction,
            confidence=0.5,
            strategy_id=self.strategy_id,
        )


def _make_ohlcv_series(
    symbol: str = "AAPL",
    n: int = 20,
    start_price: float = 150.0,
    trend: float = 0.5,
) -> list[OHLCV]:
    """Generate a series of OHLCV bars with a trend."""
    bars: list[OHLCV] = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = start_price

    for i in range(n):
        price += trend
        bars.append(OHLCV(
            timestamp=base + timedelta(hours=i),
            symbol=symbol,
            open=Decimal(str(round(price - 0.5, 2))),
            high=Decimal(str(round(price + 1.0, 2))),
            low=Decimal(str(round(price - 1.0, 2))),
            close=Decimal(str(round(price, 2))),
            volume=Decimal("1000000"),
        ))
    return bars


class TestBacktestConfig:
    """Tests for backtest configuration."""

    def test_default_config(self) -> None:
        """Config should have sensible defaults."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            symbol="AAPL",
        )
        assert config.initial_capital == Decimal("100000")
        assert config.commission_per_trade == Decimal("0")
        assert config.slippage_bps == 0
        assert config.position_size_pct == Decimal("0.1")

    def test_custom_config(self) -> None:
        """Config should accept custom parameters."""
        config = BacktestConfig(
            initial_capital=Decimal("50000"),
            symbol="TSLA",
            commission_per_trade=Decimal("1.0"),
            slippage_bps=5,
            position_size_pct=Decimal("0.05"),
        )
        assert config.initial_capital == Decimal("50000")
        assert config.commission_per_trade == Decimal("1.0")


class TestBacktestEngine:
    """Tests for backtest engine."""

    def test_engine_initialization(self) -> None:
        """Engine should initialize with strategy and config."""
        strategy = SimpleTestStrategy("test_1", {})
        config = BacktestConfig(initial_capital=Decimal("100000"), symbol="AAPL")
        engine = BacktestEngine(strategy=strategy, config=config)
        assert engine.strategy is strategy

    def test_run_backtest_returns_result(self) -> None:
        """Backtest should return a BacktestResult."""
        strategy = SimpleTestStrategy("test_1", {})
        config = BacktestConfig(initial_capital=Decimal("100000"), symbol="AAPL")
        engine = BacktestEngine(strategy=strategy, config=config)

        bars = _make_ohlcv_series(n=20, trend=0.5)
        result = engine.run(bars)

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == Decimal("100000")
        assert len(result.equity_curve) > 0

    def test_deterministic_replay(self) -> None:
        """Same data should produce same results (deterministic)."""
        bars = _make_ohlcv_series(n=20, trend=0.5)

        strategy1 = SimpleTestStrategy("test_1", {})
        config = BacktestConfig(initial_capital=Decimal("100000"), symbol="AAPL")
        result1 = BacktestEngine(strategy=strategy1, config=config).run(bars)

        strategy2 = SimpleTestStrategy("test_1", {})
        result2 = BacktestEngine(strategy=strategy2, config=config).run(bars)

        assert result1.equity_curve == result2.equity_curve
        assert result1.total_trades == result2.total_trades

    def test_equity_curve_length(self) -> None:
        """Equity curve should have one entry per bar."""
        strategy = SimpleTestStrategy("test_1", {})
        config = BacktestConfig(initial_capital=Decimal("100000"), symbol="AAPL")
        engine = BacktestEngine(strategy=strategy, config=config)

        bars = _make_ohlcv_series(n=15)
        result = engine.run(bars)
        assert len(result.equity_curve) == 15

    def test_uptrend_positive_pnl(self) -> None:
        """Long-only strategy in uptrend should have positive PnL."""
        strategy = SimpleTestStrategy("test_1", {})
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            symbol="AAPL",
            position_size_pct=Decimal("0.1"),
        )
        engine = BacktestEngine(strategy=strategy, config=config)

        bars = _make_ohlcv_series(n=30, trend=1.0)
        result = engine.run(bars)
        assert result.final_value >= result.initial_capital

    def test_result_contains_metrics(self) -> None:
        """Result should include performance metrics."""
        strategy = SimpleTestStrategy("test_1", {})
        config = BacktestConfig(initial_capital=Decimal("100000"), symbol="AAPL")
        engine = BacktestEngine(strategy=strategy, config=config)

        bars = _make_ohlcv_series(n=20)
        result = engine.run(bars)

        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "total_trades" in result.metrics

    def test_commission_reduces_returns(self) -> None:
        """Commissions should reduce portfolio value."""
        bars = _make_ohlcv_series(n=20, trend=0.5)

        strategy_no_comm = SimpleTestStrategy("test_1", {})
        config_no_comm = BacktestConfig(
            initial_capital=Decimal("100000"),
            symbol="AAPL",
            commission_per_trade=Decimal("0"),
        )
        result_no_comm = BacktestEngine(strategy=strategy_no_comm, config=config_no_comm).run(bars)

        strategy_comm = SimpleTestStrategy("test_1", {})
        config_comm = BacktestConfig(
            initial_capital=Decimal("100000"),
            symbol="AAPL",
            commission_per_trade=Decimal("10"),
        )
        result_comm = BacktestEngine(strategy=strategy_comm, config=config_comm).run(bars)

        assert result_comm.final_value <= result_no_comm.final_value

    def test_trade_log_populated(self) -> None:
        """Backtest should record trade history."""
        strategy = SimpleTestStrategy("test_1", {})
        config = BacktestConfig(initial_capital=Decimal("100000"), symbol="AAPL")
        engine = BacktestEngine(strategy=strategy, config=config)

        bars = _make_ohlcv_series(n=20)
        result = engine.run(bars)
        assert result.total_trades >= 0
        assert isinstance(result.trade_log, list)
