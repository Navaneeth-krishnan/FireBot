"""Tests for Paper Trading Engine and Portfolio Simulator - TDD RED phase."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from firebot.core.models import Order, OrderSide, OrderType, Signal, SignalDirection
from firebot.execution.engine import PaperTradingEngine
from firebot.execution.portfolio import PortfolioSimulator


class TestPaperTradingEngine:
    """Tests for the Paper Trading Engine."""

    @pytest.fixture
    def engine(self) -> PaperTradingEngine:
        """Create a paper trading engine instance."""
        return PaperTradingEngine(fill_model="instant")

    def test_engine_initialization(self, engine: PaperTradingEngine) -> None:
        """Engine should initialize with default settings."""
        assert engine.fill_model == "instant"
        assert engine.slippage_bps == 0

    def test_submit_market_order(self, engine: PaperTradingEngine) -> None:
        """Engine should accept and process market orders."""
        order = Order(
            id="order_001",
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            strategy_id="test_strat",
        )
        result = engine.submit_order(order, current_price=Decimal("150.00"))
        assert result.status == "filled"
        assert result.fill_price is not None

    def test_instant_fill_at_current_price(self, engine: PaperTradingEngine) -> None:
        """Instant fill should execute at current market price."""
        order = Order(
            id="order_002",
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("50"),
            strategy_id="test_strat",
        )
        result = engine.submit_order(order, current_price=Decimal("100.00"))
        assert result.fill_price == Decimal("100.00")

    def test_order_with_slippage(self) -> None:
        """Engine should apply slippage when configured."""
        engine = PaperTradingEngine(fill_model="instant", slippage_bps=10)  # 10 bps
        order = Order(
            id="order_003",
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            strategy_id="test_strat",
        )
        result = engine.submit_order(order, current_price=Decimal("100.00"))
        # Buy with 10bps slippage: 100 * 1.001 = 100.10
        assert result.fill_price == Decimal("100.10")

    def test_sell_order_slippage_favorable(self) -> None:
        """Sell orders should have unfavorable slippage (lower price)."""
        engine = PaperTradingEngine(fill_model="instant", slippage_bps=10)
        order = Order(
            id="order_004",
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            strategy_id="test_strat",
        )
        result = engine.submit_order(order, current_price=Decimal("100.00"))
        # Sell with 10bps slippage: 100 * 0.999 = 99.90
        assert result.fill_price == Decimal("99.90")

    def test_signal_to_order_conversion(self, engine: PaperTradingEngine) -> None:
        """Engine should convert signals to orders."""
        signal = Signal(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            direction=SignalDirection.LONG,
            confidence=0.8,
            strategy_id="test_strat",
        )
        order = engine.signal_to_order(
            signal,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
        )
        assert order.side == OrderSide.BUY
        assert order.symbol == "AAPL"
        assert order.quantity == Decimal("100")


class TestPortfolioSimulator:
    """Tests for the Portfolio Simulator."""

    @pytest.fixture
    def portfolio(self) -> PortfolioSimulator:
        """Create a portfolio simulator instance."""
        return PortfolioSimulator(
            strategy_id="test_strat",
            initial_capital=Decimal("100000.00"),
        )

    def test_portfolio_initialization(self, portfolio: PortfolioSimulator) -> None:
        """Portfolio should initialize with cash and no positions."""
        assert portfolio.cash == Decimal("100000.00")
        assert portfolio.total_value == Decimal("100000.00")
        assert len(portfolio.positions) == 0

    def test_open_long_position(self, portfolio: PortfolioSimulator) -> None:
        """Portfolio should track new long positions."""
        portfolio.execute_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == Decimal("100")
        # Cash reduced: 100000 - (100 * 150) = 85000
        assert portfolio.cash == Decimal("85000.00")

    def test_close_position(self, portfolio: PortfolioSimulator) -> None:
        """Portfolio should close positions and update PnL."""
        # Open position
        portfolio.execute_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )
        # Close position at higher price
        portfolio.execute_fill(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            price=Decimal("160.00"),
        )
        # Position should be closed
        assert "AAPL" not in portfolio.positions
        # Cash: 85000 + (100 * 160) = 101000
        assert portfolio.cash == Decimal("101000.00")
        # Realized PnL: (160 - 150) * 100 = 1000
        assert portfolio.realized_pnl == Decimal("1000.00")

    def test_partial_position_close(self, portfolio: PortfolioSimulator) -> None:
        """Portfolio should handle partial position closes."""
        portfolio.execute_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )
        portfolio.execute_fill(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=Decimal("50"),
            price=Decimal("160.00"),
        )
        # 50 shares remaining
        assert portfolio.positions["AAPL"].quantity == Decimal("50")

    def test_total_value_with_positions(self, portfolio: PortfolioSimulator) -> None:
        """Total value should include cash and position values."""
        portfolio.execute_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )
        portfolio.update_price("AAPL", Decimal("160.00"))
        # Cash: 85000, Position value: 100 * 160 = 16000
        assert portfolio.total_value == Decimal("101000.00")

    def test_drawdown_calculation(self, portfolio: PortfolioSimulator) -> None:
        """Portfolio should track drawdown from high water mark."""
        # Initial HWM is 100000
        assert portfolio.high_water_mark == Decimal("100000.00")

        # Simulate loss
        portfolio.execute_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )
        portfolio.update_price("AAPL", Decimal("140.00"))  # 10 loss per share
        # Total: 85000 + 14000 = 99000
        # Drawdown: (100000 - 99000) / 100000 = 1%
        assert portfolio.drawdown == Decimal("0.01")

    def test_high_water_mark_updates(self, portfolio: PortfolioSimulator) -> None:
        """High water mark should update on new highs."""
        portfolio.execute_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )
        portfolio.update_price("AAPL", Decimal("170.00"))  # New high
        # Total: 85000 + 17000 = 102000
        portfolio.update_high_water_mark()
        assert portfolio.high_water_mark == Decimal("102000.00")


class TestRiskControls:
    """Tests for risk management controls."""

    def test_max_drawdown_breach_detection(self) -> None:
        """Portfolio should detect max drawdown breach."""
        portfolio = PortfolioSimulator(
            strategy_id="test_strat",
            initial_capital=Decimal("100000.00"),
            max_drawdown_pct=Decimal("0.10"),  # 10%
        )
        # Simulate 15% loss
        portfolio.execute_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("1000"),
            price=Decimal("100.00"),
        )
        portfolio.update_price("AAPL", Decimal("85.00"))  # 15% loss on position

        assert portfolio.is_drawdown_breached() is True

    def test_position_size_limit(self) -> None:
        """Portfolio should enforce position size limits."""
        portfolio = PortfolioSimulator(
            strategy_id="test_strat",
            initial_capital=Decimal("100000.00"),
            max_position_size_pct=Decimal("0.05"),  # 5%
        )
        # Try to buy 10% of portfolio
        max_qty = portfolio.calculate_max_position_size(
            symbol="AAPL",
            price=Decimal("100.00"),
        )
        # Max 5% = 5000, at 100/share = 50 shares
        assert max_qty == Decimal("50")
