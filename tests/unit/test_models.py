"""Tests for core data models - TDD RED phase."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from firebot.core.models import (
    OHLCV,
    Order,
    OrderSide,
    OrderType,
    Position,
    Portfolio,
    Signal,
    SignalDirection,
)


class TestOHLCV:
    """Tests for OHLCV data model."""

    def test_create_ohlcv_with_valid_data(self) -> None:
        """OHLCV should be created with valid OHLCV data."""
        ohlcv = OHLCV(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("185.50"),
            high=Decimal("186.25"),
            low=Decimal("185.00"),
            close=Decimal("186.00"),
            volume=Decimal("1000000"),
            resolution="1h",
        )
        assert ohlcv.symbol == "AAPL"
        assert ohlcv.close == Decimal("186.00")
        assert ohlcv.resolution == "1h"

    def test_ohlcv_validates_high_gte_low(self) -> None:
        """OHLCV should validate that high >= low."""
        with pytest.raises(ValueError, match="high must be >= low"):
            OHLCV(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                symbol="AAPL",
                open=Decimal("185.50"),
                high=Decimal("184.00"),  # Invalid: high < low
                low=Decimal("185.00"),
                close=Decimal("186.00"),
                volume=Decimal("1000000"),
                resolution="1h",
            )

    def test_ohlcv_validates_positive_volume(self) -> None:
        """OHLCV should validate that volume >= 0."""
        with pytest.raises(ValueError):
            OHLCV(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                symbol="AAPL",
                open=Decimal("185.50"),
                high=Decimal("186.25"),
                low=Decimal("185.00"),
                close=Decimal("186.00"),
                volume=Decimal("-100"),  # Invalid: negative volume
                resolution="1h",
            )


class TestSignal:
    """Tests for Signal data model."""

    def test_create_long_signal(self) -> None:
        """Signal should represent a long position recommendation."""
        signal = Signal(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            direction=SignalDirection.LONG,
            confidence=0.85,
            strategy_id="momentum_v1",
        )
        assert signal.direction == SignalDirection.LONG
        assert signal.confidence == 0.85
        assert signal.strategy_id == "momentum_v1"

    def test_signal_validates_confidence_range(self) -> None:
        """Signal confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                symbol="AAPL",
                direction=SignalDirection.LONG,
                confidence=1.5,  # Invalid: > 1
                strategy_id="momentum_v1",
            )

    def test_signal_with_metadata(self) -> None:
        """Signal can contain optional metadata."""
        signal = Signal(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            confidence=0.75,
            strategy_id="ml_v1",
            metadata={"model_version": "1.2.0", "features_used": ["sma_20", "rsi"]},
        )
        assert signal.metadata["model_version"] == "1.2.0"


class TestOrder:
    """Tests for Order data model."""

    def test_create_market_order(self) -> None:
        """Order should represent a market buy order."""
        order = Order(
            id="order_001",
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            strategy_id="momentum_v1",
        )
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.price is None  # Market orders have no price

    def test_create_limit_order(self) -> None:
        """Order should represent a limit sell order with price."""
        order = Order(
            id="order_002",
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("50"),
            price=Decimal("190.00"),
            strategy_id="momentum_v1",
        )
        assert order.side == OrderSide.SELL
        assert order.price == Decimal("190.00")

    def test_order_validates_positive_quantity(self) -> None:
        """Order quantity must be positive."""
        with pytest.raises(ValueError):
            Order(
                id="order_003",
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("-10"),  # Invalid
                strategy_id="momentum_v1",
            )


class TestPosition:
    """Tests for Position data model."""

    def test_create_position(self) -> None:
        """Position should track current holdings."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("185.00"),
            current_price=Decimal("190.00"),
            strategy_id="momentum_v1",
        )
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")

    def test_position_unrealized_pnl(self) -> None:
        """Position should calculate unrealized PnL correctly."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("185.00"),
            current_price=Decimal("190.00"),
            strategy_id="momentum_v1",
        )
        # Unrealized PnL = (current - entry) * quantity = (190 - 185) * 100 = 500
        assert position.unrealized_pnl == Decimal("500.00")

    def test_position_unrealized_pnl_negative(self) -> None:
        """Position should calculate negative unrealized PnL."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("190.00"),
            current_price=Decimal("185.00"),
            strategy_id="momentum_v1",
        )
        # Unrealized PnL = (185 - 190) * 100 = -500
        assert position.unrealized_pnl == Decimal("-500.00")


class TestPortfolio:
    """Tests for Portfolio data model."""

    def test_create_portfolio(self) -> None:
        """Portfolio should track aggregate state."""
        portfolio = Portfolio(
            strategy_id="momentum_v1",
            cash=Decimal("100000.00"),
            positions={},
        )
        assert portfolio.cash == Decimal("100000.00")
        assert portfolio.total_value == Decimal("100000.00")

    def test_portfolio_with_positions(self) -> None:
        """Portfolio total value includes position values."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("185.00"),
            current_price=Decimal("190.00"),
            strategy_id="momentum_v1",
        )
        portfolio = Portfolio(
            strategy_id="momentum_v1",
            cash=Decimal("81500.00"),  # 100000 - 185*100
            positions={"AAPL": position},
        )
        # Total = cash + position_value = 81500 + (190 * 100) = 100500
        assert portfolio.total_value == Decimal("100500.00")

    def test_portfolio_drawdown_calculation(self) -> None:
        """Portfolio should track max drawdown."""
        portfolio = Portfolio(
            strategy_id="momentum_v1",
            cash=Decimal("90000.00"),
            positions={},
            high_water_mark=Decimal("100000.00"),
        )
        # Drawdown = (high_water_mark - current) / high_water_mark
        # = (100000 - 90000) / 100000 = 0.10 = 10%
        assert portfolio.drawdown == Decimal("0.10")
