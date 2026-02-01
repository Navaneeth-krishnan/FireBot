"""Backtesting engine for deterministic strategy replay.

Provides a complete backtesting loop: iterates over historical bars,
feeds them to a strategy, generates signals, executes orders via the
paper trading engine, and collects performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from firebot.core.models import OHLCV, Order, OrderSide, OrderType, SignalDirection
from firebot.execution.engine import PaperTradingEngine
from firebot.execution.portfolio import PortfolioSimulator
from firebot.metrics.calculators import (
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
)
from firebot.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a backtest run.

    Attributes:
        initial_capital: Starting cash
        symbol: Symbol to trade
        commission_per_trade: Per-trade commission
        slippage_bps: Slippage in basis points
        position_size_pct: Fraction of portfolio per trade (0-1)
    """

    initial_capital: Decimal
    symbol: str
    commission_per_trade: Decimal = Decimal("0")
    slippage_bps: int = 0
    position_size_pct: Decimal = Decimal("0.1")


@dataclass
class TradeLogEntry:
    """Record of a single trade during backtest."""

    timestamp: Any
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    commission: Decimal


@dataclass
class BacktestResult:
    """Complete results from a backtest run.

    Attributes:
        initial_capital: Starting capital
        final_value: Final portfolio value
        equity_curve: List of portfolio values per bar
        total_trades: Number of trades executed
        trade_log: List of trade records
        metrics: Performance metrics dict
    """

    initial_capital: Decimal
    final_value: Decimal
    equity_curve: list[Decimal]
    total_trades: int
    trade_log: list[TradeLogEntry]
    metrics: dict[str, Any]


class BacktestEngine:
    """Deterministic backtesting engine.

    Replays historical OHLCV bars through a strategy, executing
    signals via PaperTradingEngine and tracking portfolio state.

    The backtest loop for each bar:
    1. Feed bar to strategy via on_data()
    2. Generate signal via generate_signal()
    3. If signal is directional (LONG/SHORT), calculate order size
    4. Submit order to paper trading engine
    5. Execute fill in portfolio simulator
    6. Record equity snapshot

    Example:
        strategy = MomentumStrategy("mom_1", {"lookback_window": 20})
        config = BacktestConfig(initial_capital=Decimal("100000"), symbol="AAPL")
        engine = BacktestEngine(strategy=strategy, config=config)
        result = engine.run(historical_bars)
        print(result.metrics["sharpe_ratio"])
    """

    def __init__(self, strategy: Strategy, config: BacktestConfig) -> None:
        self.strategy = strategy
        self.config = config

    def run(self, bars: list[OHLCV]) -> BacktestResult:
        """Run backtest over a list of OHLCV bars.

        Args:
            bars: Historical bars in chronological order

        Returns:
            BacktestResult with equity curve and metrics
        """
        trading_engine = PaperTradingEngine(
            fill_model="instant",
            slippage_bps=self.config.slippage_bps,
            commission_per_trade=float(self.config.commission_per_trade),
        )
        portfolio = PortfolioSimulator(
            strategy_id=self.strategy.strategy_id,
            initial_capital=self.config.initial_capital,
        )

        equity_curve: list[Decimal] = []
        trade_log: list[TradeLogEntry] = []
        total_trades = 0
        order_counter = 0

        for bar in bars:
            # 1. Feed data to strategy
            self.strategy.on_data(bar)

            # 2. Update existing position prices
            portfolio.update_price(bar.symbol, bar.close)

            # 3. Check pending conditional orders
            current_prices = {bar.symbol: bar.close}
            pending_fills = trading_engine.check_pending_orders(current_prices)
            for fill_result in pending_fills:
                if fill_result.status == "filled" and fill_result.fill_price is not None:
                    # Look up the order to get the side
                    order_side = self._find_order_side(
                        trading_engine, fill_result.order_id
                    )
                    if order_side is not None and fill_result.fill_quantity is not None:
                        portfolio.execute_fill(
                            symbol=bar.symbol,
                            side=order_side,
                            quantity=fill_result.fill_quantity,
                            price=fill_result.fill_price,
                        )
                        trade_log.append(TradeLogEntry(
                            timestamp=bar.timestamp,
                            symbol=bar.symbol,
                            side=order_side.value,
                            quantity=fill_result.fill_quantity,
                            price=fill_result.fill_price,
                            commission=self.config.commission_per_trade,
                        ))
                        total_trades += 1

            # 4. Generate signal
            features = self._extract_features(bar)
            signal = self.strategy.generate_signal(features)

            # 5. If actionable signal, create and submit order
            if signal is not None and signal.direction != SignalDirection.NEUTRAL:
                order_counter += 1
                side = (
                    OrderSide.BUY
                    if signal.direction == SignalDirection.LONG
                    else OrderSide.SELL
                )

                # Calculate position size
                quantity = self._calculate_quantity(
                    portfolio.cash, bar.close, self.config.position_size_pct
                )

                if quantity > 0:
                    order = Order(
                        id=f"bt_{order_counter}",
                        timestamp=bar.timestamp,
                        symbol=bar.symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        price=bar.close,
                        strategy_id=self.strategy.strategy_id,
                    )

                    fill_result = trading_engine.submit_order(order, bar.close)

                    if fill_result.status == "filled" and fill_result.fill_price is not None:
                        portfolio.execute_fill(
                            symbol=bar.symbol,
                            side=side,
                            quantity=fill_result.fill_quantity or quantity,
                            price=fill_result.fill_price,
                        )
                        trade_log.append(TradeLogEntry(
                            timestamp=bar.timestamp,
                            symbol=bar.symbol,
                            side=side.value,
                            quantity=fill_result.fill_quantity or quantity,
                            price=fill_result.fill_price,
                            commission=self.config.commission_per_trade,
                        ))
                        total_trades += 1

            # 6. Record equity
            equity_curve.append(portfolio.total_value)

        # Calculate metrics
        final_value = portfolio.total_value
        total_commission = self.config.commission_per_trade * total_trades

        returns = calculate_returns(equity_curve)
        metrics: dict[str, Any] = {
            "sharpe_ratio": calculate_sharpe_ratio(returns),
            "max_drawdown": calculate_max_drawdown(equity_curve),
            "total_trades": total_trades,
            "total_commission": float(total_commission),
            "initial_capital": float(self.config.initial_capital),
            "final_value": float(final_value),
            "total_return": (
                float((final_value - self.config.initial_capital) / self.config.initial_capital)
                if self.config.initial_capital > 0
                else 0.0
            ),
        }

        return BacktestResult(
            initial_capital=self.config.initial_capital,
            final_value=final_value,
            equity_curve=equity_curve,
            total_trades=total_trades,
            trade_log=trade_log,
            metrics=metrics,
        )

    @staticmethod
    def _find_order_side(
        engine: PaperTradingEngine, order_id: str
    ) -> OrderSide | None:
        """Find the side of an order from the trading engine history."""
        for order, _ in engine.order_history:
            if order.id == order_id:
                return order.side
        return None

    @staticmethod
    def _calculate_quantity(
        cash: Decimal,
        price: Decimal,
        position_size_pct: Decimal,
    ) -> Decimal:
        """Calculate order quantity based on position sizing.

        Args:
            cash: Available cash
            price: Current price
            position_size_pct: Fraction of cash to use

        Returns:
            Quantity to trade (whole units)
        """
        if price <= 0:
            return Decimal("0")
        allocation = cash * position_size_pct
        quantity = allocation / price
        return Decimal(str(int(quantity)))

    @staticmethod
    def _extract_features(bar: OHLCV) -> dict[str, Any]:
        """Extract basic features from an OHLCV bar.

        Args:
            bar: Current OHLCV bar

        Returns:
            Feature dictionary
        """
        return {
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "returns": 0.0,
        }
