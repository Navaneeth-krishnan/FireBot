"""Metrics exporters for Prometheus and time-series storage."""

from decimal import Decimal
from typing import Any

from prometheus_client import Gauge, Counter, Histogram, CollectorRegistry, generate_latest


class PrometheusExporter:
    """Export trading metrics to Prometheus format.

    Creates Prometheus gauges and counters for key trading metrics,
    enabling Grafana dashboards and alerting.

    Example:
        exporter = PrometheusExporter(strategy_id="momentum_1")
        exporter.update_portfolio(cash=Decimal("100000"), total_value=Decimal("102000"))
        exporter.record_trade(pnl=100.0, side="buy")
        output = exporter.generate_metrics()
    """

    def __init__(
        self,
        strategy_id: str,
        registry: CollectorRegistry | None = None,
    ) -> None:
        """Initialize Prometheus exporter.

        Args:
            strategy_id: Strategy identifier used as label
            registry: Optional custom registry (default creates new one)
        """
        self.strategy_id = strategy_id
        self.registry = registry or CollectorRegistry()

        # Portfolio gauges
        self.portfolio_value = Gauge(
            "firebot_portfolio_value",
            "Current total portfolio value",
            ["strategy_id"],
            registry=self.registry,
        )
        self.portfolio_cash = Gauge(
            "firebot_portfolio_cash",
            "Current cash balance",
            ["strategy_id"],
            registry=self.registry,
        )
        self.portfolio_drawdown = Gauge(
            "firebot_portfolio_drawdown",
            "Current drawdown from high water mark",
            ["strategy_id"],
            registry=self.registry,
        )
        self.portfolio_high_water_mark = Gauge(
            "firebot_portfolio_hwm",
            "Portfolio high water mark",
            ["strategy_id"],
            registry=self.registry,
        )

        # Performance gauges
        self.sharpe_ratio = Gauge(
            "firebot_sharpe_ratio",
            "Current Sharpe ratio",
            ["strategy_id"],
            registry=self.registry,
        )
        self.sortino_ratio = Gauge(
            "firebot_sortino_ratio",
            "Current Sortino ratio",
            ["strategy_id"],
            registry=self.registry,
        )
        self.win_rate = Gauge(
            "firebot_win_rate",
            "Current win rate",
            ["strategy_id"],
            registry=self.registry,
        )
        self.profit_factor = Gauge(
            "firebot_profit_factor",
            "Current profit factor",
            ["strategy_id"],
            registry=self.registry,
        )

        # Trade counters
        self.trades_total = Counter(
            "firebot_trades_total",
            "Total number of trades",
            ["strategy_id", "side"],
            registry=self.registry,
        )
        self.trade_pnl = Histogram(
            "firebot_trade_pnl",
            "Trade PnL distribution",
            ["strategy_id"],
            buckets=[-1000, -500, -100, -50, 0, 50, 100, 500, 1000, 5000],
            registry=self.registry,
        )

    def update_portfolio(
        self,
        cash: Decimal,
        total_value: Decimal,
        drawdown: Decimal = Decimal("0"),
        high_water_mark: Decimal = Decimal("0"),
    ) -> None:
        """Update portfolio metrics.

        Args:
            cash: Current cash balance
            total_value: Total portfolio value
            drawdown: Current drawdown
            high_water_mark: Portfolio high water mark
        """
        self.portfolio_value.labels(strategy_id=self.strategy_id).set(float(total_value))
        self.portfolio_cash.labels(strategy_id=self.strategy_id).set(float(cash))
        self.portfolio_drawdown.labels(strategy_id=self.strategy_id).set(float(drawdown))
        self.portfolio_high_water_mark.labels(strategy_id=self.strategy_id).set(
            float(high_water_mark)
        )

    def update_performance(
        self,
        sharpe: float = 0.0,
        sortino: float = 0.0,
        win_rate_val: float = 0.0,
        profit_factor_val: float = 0.0,
    ) -> None:
        """Update performance metrics.

        Args:
            sharpe: Sharpe ratio
            sortino: Sortino ratio
            win_rate_val: Win rate (0-1)
            profit_factor_val: Profit factor
        """
        self.sharpe_ratio.labels(strategy_id=self.strategy_id).set(sharpe)
        self.sortino_ratio.labels(strategy_id=self.strategy_id).set(sortino)
        self.win_rate.labels(strategy_id=self.strategy_id).set(win_rate_val)
        self.profit_factor.labels(strategy_id=self.strategy_id).set(profit_factor_val)

    def record_trade(self, pnl: float, side: str) -> None:
        """Record a completed trade.

        Args:
            pnl: Trade profit/loss
            side: Trade side ("buy" or "sell")
        """
        self.trades_total.labels(strategy_id=self.strategy_id, side=side).inc()
        self.trade_pnl.labels(strategy_id=self.strategy_id).observe(pnl)

    def generate_metrics(self) -> bytes:
        """Generate Prometheus exposition format output.

        Returns:
            Bytes containing Prometheus metrics text
        """
        return generate_latest(self.registry)
