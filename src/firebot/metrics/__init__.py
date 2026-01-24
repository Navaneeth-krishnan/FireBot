"""Metrics engine for performance analytics."""

from firebot.metrics.calculators import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
)
from firebot.metrics.engine import MetricsEngine

__all__ = [
    "calculate_returns",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_win_rate",
    "calculate_profit_factor",
    "MetricsEngine",
]
