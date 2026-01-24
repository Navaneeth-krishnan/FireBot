"""Metrics Engine for performance tracking and analytics."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from firebot.metrics.calculators import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
)


@dataclass
class EquitySnapshot:
    """Equity value at a point in time."""

    value: Decimal
    timestamp: datetime


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    symbol: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    side: str  # "long" or "short"
    pnl: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now())


class MetricsEngine:
    """Engine for calculating and tracking performance metrics.

    Supports:
    - Equity curve tracking
    - Trade recording
    - Sharpe ratio calculation
    - Sortino ratio calculation
    - Maximum drawdown tracking
    - Win rate calculation
    - Profit factor calculation

    Example:
        engine = MetricsEngine()
        engine.record_equity(Decimal("100000"), datetime.now())
        engine.record_trade("AAPL", Decimal("150"), Decimal("160"), Decimal("100"), "long")
        metrics = engine.calculate_metrics()
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """Initialize metrics engine.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
        """
        self._equity_snapshots: list[EquitySnapshot] = []
        self._trades: list[TradeRecord] = []
        self.risk_free_rate = risk_free_rate

    @property
    def equity_curve(self) -> list[Decimal]:
        """Get equity values from snapshots."""
        return [s.value for s in self._equity_snapshots]

    @property
    def trades(self) -> list[dict[str, Any]]:
        """Get trades as list of dicts for compatibility."""
        return [
            {
                "symbol": t.symbol,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "side": t.side,
                "pnl": t.pnl,
                "timestamp": t.timestamp,
            }
            for t in self._trades
        ]

    def record_equity(self, value: Decimal, timestamp: datetime) -> None:
        """Record an equity snapshot.

        Args:
            value: Current portfolio value
            timestamp: Timestamp of the snapshot
        """
        self._equity_snapshots.append(EquitySnapshot(value=value, timestamp=timestamp))

    def record_trade(
        self,
        symbol: str,
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: Decimal,
        side: str,
    ) -> None:
        """Record a completed trade.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            side: "long" or "short"
        """
        if side == "long":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        self._trades.append(
            TradeRecord(
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                side=side,
                pnl=pnl,
            )
        )

    def calculate_metrics(self) -> dict[str, Any]:
        """Calculate all performance metrics.

        Returns:
            Dictionary containing all metrics
        """
        returns = calculate_returns(self.equity_curve)
        trades_list = self.trades

        total_pnl = sum(t["pnl"] for t in trades_list) if trades_list else Decimal("0")

        return {
            "sharpe_ratio": calculate_sharpe_ratio(returns, self.risk_free_rate),
            "sortino_ratio": calculate_sortino_ratio(returns, self.risk_free_rate),
            "max_drawdown": calculate_max_drawdown(self.equity_curve),
            "win_rate": calculate_win_rate(trades_list),
            "profit_factor": calculate_profit_factor(trades_list),
            "total_trades": len(trades_list),
            "total_pnl": total_pnl,
            "returns_count": len(returns),
            "equity_snapshots": len(self._equity_snapshots),
        }

    def get_summary_report(self) -> str:
        """Generate a human-readable summary report.

        Returns:
            Formatted string with metrics summary
        """
        metrics = self.calculate_metrics()

        lines = [
            "=" * 50,
            "PERFORMANCE SUMMARY",
            "=" * 50,
            "",
            "Risk-Adjusted Returns:",
            f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}",
            f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}",
            "",
            "Risk Metrics:",
            f"  Max Drawdown:  {metrics['max_drawdown']:.2%}",
            "",
            "Trade Statistics:",
            f"  Total Trades:  {metrics['total_trades']}",
            f"  Win Rate:      {metrics['win_rate']:.2%}",
            f"  Profit Factor: {metrics['profit_factor']:.2f}",
            f"  Total PnL:     ${float(metrics['total_pnl']):,.2f}",
            "",
            "=" * 50,
        ]

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all recorded data."""
        self._equity_snapshots = []
        self._trades = []
