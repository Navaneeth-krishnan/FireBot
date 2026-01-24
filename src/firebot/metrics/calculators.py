"""Performance metric calculators."""

import math
from decimal import Decimal
from typing import Any


def calculate_returns(equity_curve: list[Decimal]) -> list[float]:
    """Calculate simple returns from equity curve.

    Args:
        equity_curve: List of equity values over time

    Returns:
        List of period returns (as floats)
    """
    if len(equity_curve) < 2:
        return []

    returns = []
    for i in range(1, len(equity_curve)):
        prev_value = float(equity_curve[i - 1])
        curr_value = float(equity_curve[i])
        if prev_value != 0:
            ret = (curr_value - prev_value) / prev_value
            returns.append(ret)
    return returns


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (default 252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    # Calculate mean and std of returns
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_return = math.sqrt(variance)

    if std_return == 0:
        return 0.0

    # Period risk-free rate
    period_rf = risk_free_rate / periods_per_year

    # Sharpe ratio (annualized)
    excess_return = mean_return - period_rf
    sharpe = (excess_return / std_return) * math.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino ratio (uses downside deviation).

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (default 252 for daily)

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)

    # Calculate downside deviation (only negative returns)
    negative_returns = [r for r in returns if r < 0]
    if not negative_returns:
        # No downside, return a high value
        return 100.0  # Cap at reasonable value

    downside_variance = sum(r**2 for r in negative_returns) / len(returns)
    downside_std = math.sqrt(downside_variance)

    if downside_std == 0:
        return 100.0

    # Period risk-free rate
    period_rf = risk_free_rate / periods_per_year

    # Sortino ratio (annualized)
    excess_return = mean_return - period_rf
    sortino = (excess_return / downside_std) * math.sqrt(periods_per_year)

    return sortino


def calculate_max_drawdown(equity_curve: list[Decimal]) -> float:
    """Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: List of equity values over time

    Returns:
        Maximum drawdown as a decimal (0.10 = 10%)
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_drawdown = Decimal("0")

    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else Decimal("0")
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return float(max_drawdown)


def calculate_win_rate(trades: list[dict[str, Any]]) -> float:
    """Calculate win rate from trades.

    Args:
        trades: List of trade dicts with 'pnl' key

    Returns:
        Win rate as decimal (0.50 = 50%)
    """
    if not trades:
        return 0.0

    winning_trades = sum(1 for t in trades if t["pnl"] > 0)
    return winning_trades / len(trades)


def calculate_profit_factor(trades: list[dict[str, Any]]) -> float:
    """Calculate profit factor (gross profit / gross loss).

    Args:
        trades: List of trade dicts with 'pnl' key

    Returns:
        Profit factor (>1 is profitable)
    """
    if not trades:
        return 0.0

    gross_profit = sum(float(t["pnl"]) for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(float(t["pnl"]) for t in trades if t["pnl"] < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss
