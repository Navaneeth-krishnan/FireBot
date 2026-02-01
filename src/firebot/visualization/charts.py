"""Chart generators for trading visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


@dataclass(frozen=True)
class ChartConfig:
    """Configuration for chart appearance.

    Attributes:
        width: Figure width in inches
        height: Figure height in inches
        dpi: Resolution in dots per inch
        style: Matplotlib style name
    """

    width: int = 12
    height: int = 6
    dpi: int = 100
    style: str = "seaborn-v0_8-darkgrid"


class _BaseChart:
    """Base class for all chart types."""

    def __init__(self, config: ChartConfig | None = None) -> None:
        self._config = config or ChartConfig()
        self._fig: Figure | None = None

    def _create_figure(self) -> tuple[Figure, Any]:
        """Create a new matplotlib figure with configured size."""
        plt.style.use(self._config.style)
        fig, ax = plt.subplots(
            figsize=(self._config.width, self._config.height),
            dpi=self._config.dpi,
        )
        self._fig = fig
        return fig, ax

    def save(self, filepath: Path) -> None:
        """Save the current figure to a file.

        Args:
            filepath: Output path (supports PNG, SVG, PDF)
        """
        if self._fig is None:
            raise RuntimeError("No chart to save. Call plot() first.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(filepath, bbox_inches="tight", dpi=self._config.dpi)


class EquityCurveChart(_BaseChart):
    """Equity curve visualization.

    Plots portfolio value over time, supporting single or multi-strategy
    overlays with legend and grid.

    Example:
        chart = EquityCurveChart()
        fig = chart.plot(timestamps, values, title="My Equity Curve")
        chart.save(Path("equity.png"))
    """

    def plot(
        self,
        timestamps: list[datetime],
        values: list[Decimal],
        title: str = "Equity Curve",
    ) -> Figure:
        """Plot a single equity curve.

        Args:
            timestamps: Time axis data
            values: Portfolio values
            title: Chart title

        Returns:
            Matplotlib Figure

        Raises:
            ValueError: If data is empty
        """
        if not timestamps or not values:
            raise ValueError("Cannot plot empty data")

        fig, ax = self._create_figure()
        float_values = [float(v) for v in values]

        ax.plot(timestamps, float_values, linewidth=1.5, color="#2196F3")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_multiple(
        self,
        strategy_data: dict[str, dict[str, list]],
        title: str = "Strategy Equity Curves",
    ) -> Figure:
        """Plot multiple strategy equity curves overlaid.

        Args:
            strategy_data: Dict mapping strategy_id to
                {"timestamps": [...], "values": [...]}
            title: Chart title

        Returns:
            Matplotlib Figure
        """
        fig, ax = self._create_figure()
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(strategy_data), 1)))

        for idx, (strategy_id, data) in enumerate(strategy_data.items()):
            float_values = [float(v) for v in data["values"]]
            ax.plot(
                data["timestamps"],
                float_values,
                linewidth=1.5,
                color=colors[idx],
                label=strategy_id,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        ax.legend(loc="upper left")
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)

        return fig


class DrawdownChart(_BaseChart):
    """Drawdown visualization.

    Plots drawdown from high water mark as a filled area chart,
    showing periods and depth of losses.

    Example:
        chart = DrawdownChart()
        fig = chart.plot(timestamps, equity_values)
        chart.save(Path("drawdown.png"))
    """

    @staticmethod
    def compute_drawdowns(equity_values: list[Decimal]) -> list[float]:
        """Compute drawdown percentages from equity values.

        Args:
            equity_values: List of portfolio values

        Returns:
            List of drawdown percentages (positive values = drawdown)
        """
        if not equity_values:
            return []

        drawdowns: list[float] = []
        high_water_mark = float(equity_values[0])

        for value in equity_values:
            fval = float(value)
            if fval > high_water_mark:
                high_water_mark = fval
            dd_pct = ((high_water_mark - fval) / high_water_mark) * 100 if high_water_mark > 0 else 0.0
            drawdowns.append(round(dd_pct, 2))

        return drawdowns

    def plot(
        self,
        timestamps: list[datetime],
        equity_values: list[Decimal],
        title: str = "Portfolio Drawdown",
        annotate_max: bool = False,
    ) -> Figure:
        """Plot drawdown chart.

        Args:
            timestamps: Time axis data
            equity_values: Portfolio values (drawdowns computed internally)
            title: Chart title
            annotate_max: Whether to annotate the maximum drawdown point

        Returns:
            Matplotlib Figure
        """
        fig, ax = self._create_figure()
        drawdowns = self.compute_drawdowns(equity_values)
        neg_drawdowns = [-d for d in drawdowns]

        ax.fill_between(timestamps, 0, neg_drawdowns, alpha=0.4, color="#F44336")
        ax.plot(timestamps, neg_drawdowns, linewidth=1.0, color="#D32F2F")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)

        if annotate_max and drawdowns:
            max_dd_idx = drawdowns.index(max(drawdowns))
            max_dd_val = drawdowns[max_dd_idx]
            ax.annotate(
                f"Max DD: {max_dd_val:.2f}%",
                xy=(timestamps[max_dd_idx], -max_dd_val),
                xytext=(timestamps[max_dd_idx], -max_dd_val - 1),
                fontsize=10,
                fontweight="bold",
                color="#D32F2F",
                ha="center",
            )

        return fig


class StrategyComparisonChart(_BaseChart):
    """Strategy comparison visualizations.

    Provides bar charts, radar plots, and table summaries for
    comparing multiple strategy performance metrics.

    Example:
        chart = StrategyComparisonChart()
        fig = chart.plot_metric_bars(metrics, "sharpe_ratio")
        chart.save(Path("comparison.png"))
    """

    def plot_metric_bars(
        self,
        strategy_metrics: dict[str, dict[str, float]],
        metric_name: str,
        title: str | None = None,
    ) -> Figure:
        """Plot a bar chart comparing a single metric across strategies.

        Args:
            strategy_metrics: Dict mapping strategy_id to metrics dict
            metric_name: Key name of the metric to compare
            title: Optional chart title

        Returns:
            Matplotlib Figure
        """
        fig, ax = self._create_figure()
        strategy_ids = list(strategy_metrics.keys())
        values = [strategy_metrics[s][metric_name] for s in strategy_ids]
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(strategy_ids), 1)))

        bars = ax.bar(strategy_ids, values, color=colors, edgecolor="white", linewidth=0.5)

        display_title = title or metric_name.replace("_", " ").title()
        ax.set_title(display_title, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        return fig

    def plot_radar(
        self,
        strategy_metrics: dict[str, dict[str, float]],
        metrics: list[str],
        title: str = "Strategy Comparison",
    ) -> Figure:
        """Plot a radar/spider chart comparing strategies across multiple metrics.

        Args:
            strategy_metrics: Dict mapping strategy_id to metrics dict
            metrics: List of metric names to include
            title: Chart title

        Returns:
            Matplotlib Figure
        """
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(
            figsize=(self._config.width, self._config.height),
            dpi=self._config.dpi,
            subplot_kw={"projection": "polar"},
        )
        self._fig = fig

        colors = plt.cm.Set2(np.linspace(0, 1, max(len(strategy_metrics), 1)))

        for idx, (strategy_id, strat_metrics) in enumerate(strategy_metrics.items()):
            values = [strat_metrics[m] for m in metrics]
            values += values[:1]  # Close the polygon
            ax.plot(angles, values, linewidth=1.5, color=colors[idx], label=strategy_id)
            ax.fill(angles, values, alpha=0.1, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
        ax.set_title(title, fontsize=14, fontweight="bold", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        return fig

    @staticmethod
    def generate_comparison_table(
        strategy_metrics: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Generate a comparison table from strategy metrics.

        Args:
            strategy_metrics: Dict mapping strategy_id to metrics dict

        Returns:
            Same structure, suitable for rendering as a table
        """
        return {
            strategy_id: dict(metrics)
            for strategy_id, metrics in strategy_metrics.items()
        }
