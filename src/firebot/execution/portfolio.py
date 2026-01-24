"""Portfolio Simulator for tracking positions, cash, and PnL."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from firebot.core.models import OrderSide, Position


@dataclass
class Trade:
    """Record of a completed trade."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal | None = None
    pnl: Decimal = Decimal("0")
    is_closed: bool = False


class PortfolioSimulator:
    """Simulates a trading portfolio with position tracking.

    Tracks:
    - Cash balance
    - Open positions
    - Realized and unrealized PnL
    - Drawdown from high water mark
    - Trade history

    Example:
        portfolio = PortfolioSimulator(
            strategy_id="my_strat",
            initial_capital=Decimal("100000"),
        )
        portfolio.execute_fill("AAPL", OrderSide.BUY, Decimal("100"), Decimal("150"))
        portfolio.update_price("AAPL", Decimal("160"))
        print(portfolio.total_value)  # 101000
    """

    def __init__(
        self,
        strategy_id: str,
        initial_capital: Decimal,
        max_drawdown_pct: Decimal = Decimal("0.10"),
        max_position_size_pct: Decimal = Decimal("0.05"),
    ) -> None:
        """Initialize portfolio simulator.

        Args:
            strategy_id: Strategy this portfolio belongs to
            initial_capital: Starting cash amount
            max_drawdown_pct: Maximum allowed drawdown (0.10 = 10%)
            max_position_size_pct: Maximum position size as % of portfolio
        """
        self.strategy_id = strategy_id
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.high_water_mark = initial_capital
        self.realized_pnl = Decimal("0")
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_size_pct = max_position_size_pct
        self.trade_history: list[Trade] = []
        self._current_prices: dict[str, Decimal] = {}

    @property
    def total_value(self) -> Decimal:
        """Calculate total portfolio value (cash + positions)."""
        position_value = sum(
            self._get_position_value(symbol, pos)
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value

    @property
    def drawdown(self) -> Decimal:
        """Calculate current drawdown from high water mark."""
        if self.high_water_mark == Decimal("0"):
            return Decimal("0")
        current = self.total_value
        dd = (self.high_water_mark - current) / self.high_water_mark
        return dd.quantize(Decimal("0.0001"))

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized PnL from open positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def _get_position_value(self, symbol: str, position: Position) -> Decimal:
        """Get current market value of a position."""
        current_price = self._current_prices.get(symbol, position.current_price)
        return current_price * position.quantity

    def execute_fill(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
    ) -> None:
        """Execute a fill and update portfolio state.

        Args:
            symbol: Instrument symbol
            side: Buy or sell
            quantity: Number of shares/units
            price: Execution price
        """
        if side == OrderSide.BUY:
            self._open_or_add_position(symbol, quantity, price)
        else:
            self._close_or_reduce_position(symbol, quantity, price)

    def _open_or_add_position(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
    ) -> None:
        """Open a new position or add to existing."""
        cost = quantity * price
        self.cash -= cost
        self._current_prices[symbol] = price

        if symbol in self.positions:
            # Average into existing position
            existing = self.positions[symbol]
            total_qty = existing.quantity + quantity
            avg_price = (
                (existing.entry_price * existing.quantity) + (price * quantity)
            ) / total_qty
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_qty,
                entry_price=avg_price,
                current_price=price,
                realized_pnl=existing.realized_pnl,
                strategy_id=self.strategy_id,
            )
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                strategy_id=self.strategy_id,
            )

        # Record trade
        self.trade_history.append(
            Trade(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                entry_price=price,
            )
        )

    def _close_or_reduce_position(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
    ) -> None:
        """Close or reduce an existing position."""
        if symbol not in self.positions:
            raise ValueError(f"No position to sell for {symbol}")

        position = self.positions[symbol]
        if quantity > position.quantity:
            raise ValueError(
                f"Cannot sell {quantity} shares, only {position.quantity} held"
            )

        # Calculate realized PnL
        pnl = (price - position.entry_price) * quantity
        self.realized_pnl += pnl
        self.cash += quantity * price
        self._current_prices[symbol] = price

        # Update or remove position
        remaining = position.quantity - quantity
        if remaining == Decimal("0"):
            del self.positions[symbol]
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining,
                entry_price=position.entry_price,
                current_price=price,
                realized_pnl=position.realized_pnl + pnl,
                strategy_id=self.strategy_id,
            )

        # Record trade
        self.trade_history.append(
            Trade(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                entry_price=position.entry_price,
                exit_price=price,
                pnl=pnl,
                is_closed=True,
            )
        )

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update current price for a symbol.

        Args:
            symbol: Instrument symbol
            price: New price
        """
        self._current_prices[symbol] = price
        if symbol in self.positions:
            pos = self.positions[symbol]
            self.positions[symbol] = Position(
                symbol=pos.symbol,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                current_price=price,
                realized_pnl=pos.realized_pnl,
                strategy_id=pos.strategy_id,
            )

    def update_high_water_mark(self) -> None:
        """Update high water mark if current value exceeds it."""
        current = self.total_value
        if current > self.high_water_mark:
            self.high_water_mark = current

    def is_drawdown_breached(self) -> bool:
        """Check if max drawdown limit has been breached.

        Returns:
            True if current drawdown exceeds max_drawdown_pct
        """
        return self.drawdown > self.max_drawdown_pct

    def calculate_max_position_size(
        self,
        symbol: str,
        price: Decimal,
    ) -> Decimal:
        """Calculate maximum position size allowed.

        Args:
            symbol: Instrument symbol
            price: Current price

        Returns:
            Maximum number of shares/units
        """
        max_value = self.total_value * self.max_position_size_pct
        max_qty = max_value / price
        return max_qty.quantize(Decimal("1"))  # Round down to whole shares

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol.

        Args:
            symbol: Instrument symbol

        Returns:
            Position if exists, None otherwise
        """
        return self.positions.get(symbol)

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary.

        Returns:
            Dictionary with portfolio metrics
        """
        return {
            "strategy_id": self.strategy_id,
            "cash": float(self.cash),
            "total_value": float(self.total_value),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "total_pnl": float(self.realized_pnl + self.unrealized_pnl),
            "drawdown": float(self.drawdown),
            "high_water_mark": float(self.high_water_mark),
            "num_positions": len(self.positions),
            "num_trades": len(self.trade_history),
        }
