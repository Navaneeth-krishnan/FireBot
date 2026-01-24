"""Tests for Metrics Engine - TDD RED phase."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from firebot.metrics.engine import MetricsEngine
from firebot.metrics.calculators import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_returns,
)


class TestReturnsCalculation:
    """Tests for returns calculation."""

    def test_calculate_simple_returns(self) -> None:
        """Should calculate simple returns from equity curve."""
        equity_curve = [
            Decimal("100000"),
            Decimal("101000"),
            Decimal("102010"),
            Decimal("100989.90"),
        ]
        returns = calculate_returns(equity_curve)
        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.01, rel=1e-4)  # 1% gain
        assert returns[1] == pytest.approx(0.01, rel=1e-4)  # 1% gain
        assert returns[2] == pytest.approx(-0.01, rel=1e-4)  # 1% loss

    def test_empty_equity_curve(self) -> None:
        """Should return empty list for insufficient data."""
        assert calculate_returns([]) == []
        assert calculate_returns([Decimal("100000")]) == []


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_ratio_positive_returns(self) -> None:
        """Should calculate positive Sharpe for mostly positive returns."""
        # Mix of positive returns with some variance
        returns = [0.002, 0.001, 0.003, 0.001, 0.002] * 50  # ~250 days of positive returns
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        # With mostly positive returns and low vol, Sharpe should be positive
        assert sharpe > 2.0

    def test_sharpe_ratio_negative_returns(self) -> None:
        """Should calculate negative Sharpe for losses."""
        # Mix of negative returns with some variance
        returns = [-0.002, -0.001, -0.003, -0.001, -0.002] * 50  # Mostly losses
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert sharpe < 0

    def test_sharpe_ratio_empty_returns(self) -> None:
        """Should return 0 for empty returns."""
        sharpe = calculate_sharpe_ratio([])
        assert sharpe == 0.0

    def test_sharpe_ratio_annualized(self) -> None:
        """Sharpe ratio should be annualized by default."""
        # Annualization factor for daily returns = sqrt(252)
        returns = [0.001, 0.002, -0.001, 0.001, 0.003] * 50
        sharpe = calculate_sharpe_ratio(returns, periods_per_year=252)
        # Should be positive with mostly positive returns
        assert sharpe > 0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_ratio_no_downside(self) -> None:
        """Should have high Sortino with no negative returns."""
        returns = [0.01, 0.02, 0.015, 0.005, 0.01]
        sortino = calculate_sortino_ratio(returns)
        # No downside deviation, should be very high (capped)
        assert sortino > 5.0

    def test_sortino_ratio_with_downside(self) -> None:
        """Should calculate Sortino with negative returns."""
        returns = [0.01, -0.02, 0.015, -0.01, 0.01]
        sortino = calculate_sortino_ratio(returns)
        # Mix of gains and losses
        assert sortino > 0  # Net positive returns

    def test_sortino_empty_returns(self) -> None:
        """Should return 0 for empty returns."""
        sortino = calculate_sortino_ratio([])
        assert sortino == 0.0


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_max_drawdown_simple(self) -> None:
        """Should calculate max drawdown from peak."""
        equity_curve = [
            Decimal("100000"),
            Decimal("110000"),  # Peak
            Decimal("99000"),  # 10% drawdown from peak
            Decimal("105000"),
        ]
        max_dd = calculate_max_drawdown(equity_curve)
        assert max_dd == pytest.approx(0.10, rel=1e-4)  # 10% drawdown

    def test_max_drawdown_multiple_drawdowns(self) -> None:
        """Should return the maximum of multiple drawdowns."""
        equity_curve = [
            Decimal("100000"),
            Decimal("120000"),  # First peak
            Decimal("108000"),  # 10% dd
            Decimal("130000"),  # Second peak
            Decimal("104000"),  # 20% dd - this is the max
            Decimal("125000"),
        ]
        max_dd = calculate_max_drawdown(equity_curve)
        assert max_dd == pytest.approx(0.20, rel=1e-4)

    def test_max_drawdown_no_drawdown(self) -> None:
        """Should return 0 for monotonically increasing equity."""
        equity_curve = [
            Decimal("100000"),
            Decimal("110000"),
            Decimal("120000"),
        ]
        max_dd = calculate_max_drawdown(equity_curve)
        assert max_dd == 0.0

    def test_max_drawdown_empty(self) -> None:
        """Should return 0 for empty equity curve."""
        max_dd = calculate_max_drawdown([])
        assert max_dd == 0.0


class TestWinRate:
    """Tests for win rate calculation."""

    def test_win_rate_all_wins(self) -> None:
        """Should return 1.0 for all winning trades."""
        trades = [
            {"pnl": Decimal("100")},
            {"pnl": Decimal("200")},
            {"pnl": Decimal("50")},
        ]
        win_rate = calculate_win_rate(trades)
        assert win_rate == 1.0

    def test_win_rate_all_losses(self) -> None:
        """Should return 0.0 for all losing trades."""
        trades = [
            {"pnl": Decimal("-100")},
            {"pnl": Decimal("-200")},
        ]
        win_rate = calculate_win_rate(trades)
        assert win_rate == 0.0

    def test_win_rate_mixed(self) -> None:
        """Should calculate correct win rate for mixed trades."""
        trades = [
            {"pnl": Decimal("100")},  # Win
            {"pnl": Decimal("-50")},  # Loss
            {"pnl": Decimal("75")},  # Win
            {"pnl": Decimal("-25")},  # Loss
        ]
        win_rate = calculate_win_rate(trades)
        assert win_rate == 0.5  # 2/4 = 50%

    def test_win_rate_empty(self) -> None:
        """Should return 0 for no trades."""
        win_rate = calculate_win_rate([])
        assert win_rate == 0.0


class TestProfitFactor:
    """Tests for profit factor calculation."""

    def test_profit_factor_positive(self) -> None:
        """Should calculate profit factor correctly."""
        trades = [
            {"pnl": Decimal("200")},  # Win
            {"pnl": Decimal("-100")},  # Loss
            {"pnl": Decimal("150")},  # Win
            {"pnl": Decimal("-50")},  # Loss
        ]
        # Gross profit: 350, Gross loss: 150
        # Profit factor: 350 / 150 = 2.33
        pf = calculate_profit_factor(trades)
        assert pf == pytest.approx(2.333, rel=1e-2)

    def test_profit_factor_no_losses(self) -> None:
        """Should return high value when no losses."""
        trades = [
            {"pnl": Decimal("100")},
            {"pnl": Decimal("200")},
        ]
        pf = calculate_profit_factor(trades)
        assert pf == float("inf") or pf > 100  # Very high or infinite

    def test_profit_factor_no_wins(self) -> None:
        """Should return 0 when no wins."""
        trades = [
            {"pnl": Decimal("-100")},
            {"pnl": Decimal("-200")},
        ]
        pf = calculate_profit_factor(trades)
        assert pf == 0.0

    def test_profit_factor_empty(self) -> None:
        """Should return 0 for no trades."""
        pf = calculate_profit_factor([])
        assert pf == 0.0


class TestMetricsEngine:
    """Tests for the main MetricsEngine class."""

    @pytest.fixture
    def engine(self) -> MetricsEngine:
        """Create a metrics engine instance."""
        return MetricsEngine()

    def test_engine_initialization(self, engine: MetricsEngine) -> None:
        """Engine should initialize with empty state."""
        assert engine.equity_curve == []
        assert engine.trades == []

    def test_record_equity_snapshot(self, engine: MetricsEngine) -> None:
        """Engine should record equity snapshots."""
        engine.record_equity(Decimal("100000"), datetime.now(timezone.utc))
        engine.record_equity(Decimal("101000"), datetime.now(timezone.utc))
        assert len(engine.equity_curve) == 2

    def test_record_trade(self, engine: MetricsEngine) -> None:
        """Engine should record completed trades."""
        engine.record_trade(
            symbol="AAPL",
            entry_price=Decimal("150.00"),
            exit_price=Decimal("160.00"),
            quantity=Decimal("100"),
            side="long",
        )
        assert len(engine.trades) == 1
        assert engine.trades[0]["pnl"] == Decimal("1000.00")

    def test_calculate_all_metrics(self, engine: MetricsEngine) -> None:
        """Engine should calculate all metrics together."""
        # Setup equity curve
        base = Decimal("100000")
        for i in range(100):
            # Simulate some gains and losses
            change = Decimal(str(100 + (i % 10) * 10 - 50))
            base += change
            engine.record_equity(base, datetime.now(timezone.utc))

        # Add some trades
        engine.record_trade("AAPL", Decimal("150"), Decimal("160"), Decimal("10"), "long")
        engine.record_trade("AAPL", Decimal("160"), Decimal("155"), Decimal("10"), "long")
        engine.record_trade("MSFT", Decimal("300"), Decimal("310"), Decimal("5"), "long")

        metrics = engine.calculate_metrics()

        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "total_trades" in metrics
        assert "total_pnl" in metrics

    def test_get_summary_report(self, engine: MetricsEngine) -> None:
        """Engine should generate human-readable summary."""
        engine.record_equity(Decimal("100000"), datetime.now(timezone.utc))
        engine.record_equity(Decimal("105000"), datetime.now(timezone.utc))
        engine.record_trade("AAPL", Decimal("150"), Decimal("160"), Decimal("100"), "long")

        report = engine.get_summary_report()

        assert isinstance(report, str)
        assert "Sharpe" in report or "sharpe" in report.lower()
        assert "Win" in report or "win" in report.lower()
