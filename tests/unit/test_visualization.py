"""Tests for visualization module - TDD RED phase."""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from firebot.visualization.charts import (
    EquityCurveChart,
    DrawdownChart,
    StrategyComparisonChart,
    ChartConfig,
)


class TestChartConfig:
    """Tests for chart configuration."""

    def test_default_config(self) -> None:
        """Config should have sensible defaults."""
        config = ChartConfig()
        assert config.width == 12
        assert config.height == 6
        assert config.dpi == 100
        assert config.style == "seaborn-v0_8-darkgrid"

    def test_custom_config(self) -> None:
        """Config should accept custom values."""
        config = ChartConfig(width=16, height=10, dpi=150, style="ggplot")
        assert config.width == 16
        assert config.height == 10
        assert config.dpi == 150
        assert config.style == "ggplot"


class TestEquityCurveChart:
    """Tests for equity curve visualization."""

    @pytest.fixture
    def sample_data(self) -> dict:
        """Create sample equity curve data."""
        now = datetime.now(timezone.utc)
        timestamps = [now + timedelta(days=i) for i in range(10)]
        values = [
            Decimal("100000"),
            Decimal("101000"),
            Decimal("102500"),
            Decimal("101500"),
            Decimal("103000"),
            Decimal("104500"),
            Decimal("103500"),
            Decimal("105000"),
            Decimal("106000"),
            Decimal("107500"),
        ]
        return {"timestamps": timestamps, "values": values}

    def test_chart_creation(self, sample_data: dict) -> None:
        """Equity chart should create a matplotlib figure."""
        chart = EquityCurveChart()
        fig = chart.plot(
            timestamps=sample_data["timestamps"],
            values=sample_data["values"],
            title="Test Equity Curve",
        )
        assert fig is not None
        # Should have exactly one axes
        assert len(fig.axes) == 1
        fig.clear()

    def test_chart_with_custom_config(self, sample_data: dict) -> None:
        """Chart should respect custom configuration."""
        config = ChartConfig(width=16, height=10, dpi=150)
        chart = EquityCurveChart(config=config)
        fig = chart.plot(
            timestamps=sample_data["timestamps"],
            values=sample_data["values"],
        )
        # Figure size in inches
        assert fig.get_size_inches()[0] == 16
        assert fig.get_size_inches()[1] == 10
        fig.clear()

    def test_save_to_file(self, sample_data: dict, tmp_path: Path) -> None:
        """Chart should save to PNG file."""
        chart = EquityCurveChart()
        filepath = tmp_path / "equity_curve.png"
        chart.plot(
            timestamps=sample_data["timestamps"],
            values=sample_data["values"],
        )
        chart.save(filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_multi_strategy_equity_curve(self, sample_data: dict) -> None:
        """Should support plotting multiple strategy equity curves."""
        chart = EquityCurveChart()
        strategy_data = {
            "momentum_1": {
                "timestamps": sample_data["timestamps"],
                "values": sample_data["values"],
            },
            "mean_revert_1": {
                "timestamps": sample_data["timestamps"],
                "values": [v - Decimal("500") for v in sample_data["values"]],
            },
        }
        fig = chart.plot_multiple(strategy_data, title="Multi-Strategy Equity")
        assert fig is not None
        ax = fig.axes[0]
        # Should have 2 lines (one per strategy)
        assert len(ax.get_lines()) == 2
        fig.clear()

    def test_empty_data_raises(self) -> None:
        """Chart should raise ValueError for empty data."""
        chart = EquityCurveChart()
        with pytest.raises(ValueError, match="empty"):
            chart.plot(timestamps=[], values=[])


class TestDrawdownChart:
    """Tests for drawdown visualization."""

    @pytest.fixture
    def sample_equity(self) -> dict:
        """Create sample equity data with drawdown."""
        now = datetime.now(timezone.utc)
        timestamps = [now + timedelta(days=i) for i in range(8)]
        values = [
            Decimal("100000"),
            Decimal("105000"),  # New high
            Decimal("102000"),  # Drawdown: (105000-102000)/105000 = 2.86%
            Decimal("98000"),   # Drawdown: (105000-98000)/105000 = 6.67%
            Decimal("103000"),  # Recovering
            Decimal("108000"),  # New high
            Decimal("104000"),  # Drawdown: (108000-104000)/108000 = 3.70%
            Decimal("110000"),  # New high
        ]
        return {"timestamps": timestamps, "values": values}

    def test_drawdown_chart_creation(self, sample_equity: dict) -> None:
        """Drawdown chart should create a figure."""
        chart = DrawdownChart()
        fig = chart.plot(
            timestamps=sample_equity["timestamps"],
            equity_values=sample_equity["values"],
        )
        assert fig is not None
        assert len(fig.axes) == 1
        fig.clear()

    def test_drawdown_values_computed(self, sample_equity: dict) -> None:
        """Drawdown chart should compute correct drawdown percentages."""
        chart = DrawdownChart()
        drawdowns = chart.compute_drawdowns(sample_equity["values"])
        assert len(drawdowns) == len(sample_equity["values"])
        # First value has no drawdown
        assert drawdowns[0] == 0.0
        # After peak of 105000, at 98000 the drawdown should be ~6.67%
        assert drawdowns[3] == pytest.approx(6.67, abs=0.01)
        # After new peak of 108000, at 104000 the drawdown should be ~3.70%
        assert drawdowns[6] == pytest.approx(3.70, abs=0.01)

    def test_drawdown_fill_area(self, sample_equity: dict) -> None:
        """Drawdown chart should use filled area (not lines)."""
        chart = DrawdownChart()
        fig = chart.plot(
            timestamps=sample_equity["timestamps"],
            equity_values=sample_equity["values"],
        )
        ax = fig.axes[0]
        # Should have filled areas (PolyCollection)
        assert len(ax.collections) > 0
        fig.clear()

    def test_save_drawdown_chart(self, sample_equity: dict, tmp_path: Path) -> None:
        """Drawdown chart should save to file."""
        chart = DrawdownChart()
        filepath = tmp_path / "drawdown.png"
        chart.plot(
            timestamps=sample_equity["timestamps"],
            equity_values=sample_equity["values"],
        )
        chart.save(filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_max_drawdown_annotation(self, sample_equity: dict) -> None:
        """Chart should annotate the maximum drawdown point."""
        chart = DrawdownChart()
        fig = chart.plot(
            timestamps=sample_equity["timestamps"],
            equity_values=sample_equity["values"],
            annotate_max=True,
        )
        ax = fig.axes[0]
        # Should have at least one annotation
        assert len(ax.texts) > 0
        fig.clear()


class TestStrategyComparisonChart:
    """Tests for strategy comparison views."""

    @pytest.fixture
    def sample_metrics(self) -> dict:
        """Create sample metrics for multiple strategies."""
        return {
            "momentum_1": {
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.1,
                "max_drawdown": 0.08,
                "win_rate": 0.62,
                "profit_factor": 1.8,
                "total_pnl": 15000.0,
            },
            "mean_revert_1": {
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.6,
                "max_drawdown": 0.12,
                "win_rate": 0.55,
                "profit_factor": 1.4,
                "total_pnl": 8000.0,
            },
            "breakout_1": {
                "sharpe_ratio": 0.8,
                "sortino_ratio": 1.1,
                "max_drawdown": 0.15,
                "win_rate": 0.45,
                "profit_factor": 1.1,
                "total_pnl": 3000.0,
            },
        }

    def test_bar_chart_creation(self, sample_metrics: dict) -> None:
        """Strategy comparison should create a bar chart."""
        chart = StrategyComparisonChart()
        fig = chart.plot_metric_bars(
            strategy_metrics=sample_metrics,
            metric_name="sharpe_ratio",
            title="Sharpe Ratio Comparison",
        )
        assert fig is not None
        assert len(fig.axes) == 1
        fig.clear()

    def test_radar_chart_creation(self, sample_metrics: dict) -> None:
        """Strategy comparison should support radar charts."""
        chart = StrategyComparisonChart()
        fig = chart.plot_radar(
            strategy_metrics=sample_metrics,
            metrics=["sharpe_ratio", "sortino_ratio", "win_rate", "profit_factor"],
        )
        assert fig is not None
        fig.clear()

    def test_comparison_table_data(self, sample_metrics: dict) -> None:
        """Should generate comparison table data."""
        chart = StrategyComparisonChart()
        table = chart.generate_comparison_table(sample_metrics)
        assert len(table) == 3  # 3 strategies
        assert "momentum_1" in table
        assert "sharpe_ratio" in table["momentum_1"]

    def test_pnl_comparison_chart(self, sample_metrics: dict) -> None:
        """Should create PnL comparison bar chart."""
        chart = StrategyComparisonChart()
        fig = chart.plot_metric_bars(
            strategy_metrics=sample_metrics,
            metric_name="total_pnl",
            title="PnL Comparison",
        )
        ax = fig.axes[0]
        # Should have bars for each strategy
        patches = ax.patches
        assert len(patches) == 3
        fig.clear()

    def test_save_comparison(self, sample_metrics: dict, tmp_path: Path) -> None:
        """Comparison chart should save to file."""
        chart = StrategyComparisonChart()
        filepath = tmp_path / "comparison.png"
        chart.plot_metric_bars(
            strategy_metrics=sample_metrics,
            metric_name="sharpe_ratio",
        )
        chart.save(filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0
