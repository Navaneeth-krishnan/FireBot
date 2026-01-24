"""Paper Trading Engine for order simulation."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from firebot.core.models import Order, OrderSide, OrderType, Signal, SignalDirection


@dataclass
class FillResult:
    """Result of order execution."""

    order_id: str
    status: str  # "filled", "rejected", "cancelled"
    fill_price: Decimal | None
    fill_quantity: Decimal | None
    timestamp: datetime
    message: str = ""


class PaperTradingEngine:
    """Simulates order execution for paper trading.

    Supports:
    - Instant fill at current price
    - Configurable slippage
    - Order tracking

    Example:
        engine = PaperTradingEngine(fill_model="instant", slippage_bps=5)
        result = engine.submit_order(order, current_price=Decimal("100.00"))
    """

    def __init__(
        self,
        fill_model: str = "instant",
        slippage_bps: float = 0,
        commission_per_trade: float = 0,
    ) -> None:
        """Initialize paper trading engine.

        Args:
            fill_model: Fill model ("instant" or "realistic")
            slippage_bps: Slippage in basis points (100 bps = 1%)
            commission_per_trade: Fixed commission per trade
        """
        self.fill_model = fill_model
        self.slippage_bps = slippage_bps
        self.commission_per_trade = Decimal(str(commission_per_trade))
        self.order_history: list[tuple[Order, FillResult]] = []

    def submit_order(
        self,
        order: Order,
        current_price: Decimal,
    ) -> FillResult:
        """Submit an order for execution.

        Args:
            order: Order to execute
            current_price: Current market price for the symbol

        Returns:
            FillResult with execution details
        """
        if self.fill_model == "instant":
            return self._instant_fill(order, current_price)
        else:
            raise ValueError(f"Unknown fill model: {self.fill_model}")

    def _instant_fill(self, order: Order, current_price: Decimal) -> FillResult:
        """Execute order instantly at current price with slippage.

        Args:
            order: Order to fill
            current_price: Current market price

        Returns:
            FillResult with fill details
        """
        # Apply slippage
        slippage_multiplier = Decimal(str(self.slippage_bps)) / Decimal("10000")

        if order.side == OrderSide.BUY:
            # Buyer pays more (unfavorable slippage)
            fill_price = current_price * (Decimal("1") + slippage_multiplier)
        else:
            # Seller receives less (unfavorable slippage)
            fill_price = current_price * (Decimal("1") - slippage_multiplier)

        # Round to 2 decimal places
        fill_price = fill_price.quantize(Decimal("0.01"))

        result = FillResult(
            order_id=order.id,
            status="filled",
            fill_price=fill_price,
            fill_quantity=order.quantity,
            timestamp=datetime.now(timezone.utc),
        )

        self.order_history.append((order, result))
        return result

    def signal_to_order(
        self,
        signal: Signal,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Decimal | None = None,
    ) -> Order:
        """Convert a trading signal to an order.

        Args:
            signal: Signal to convert
            quantity: Order quantity
            order_type: Type of order
            price: Limit price (for limit orders)

        Returns:
            Order ready for execution

        Raises:
            ValueError: If signal direction is NEUTRAL
        """
        if signal.direction == SignalDirection.NEUTRAL:
            raise ValueError("Cannot create order from NEUTRAL signal")

        side = OrderSide.BUY if signal.direction == SignalDirection.LONG else OrderSide.SELL

        return Order(
            id=f"order_{uuid4().hex[:8]}",
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            strategy_id=signal.strategy_id,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Note: In instant fill mode, orders are filled immediately,
        so this is mainly for future fill models.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancelled, False if not found or already filled
        """
        # In instant fill mode, all orders are already filled
        return False

    def get_order_history(self) -> list[tuple[Order, FillResult]]:
        """Get complete order history.

        Returns:
            List of (order, result) tuples
        """
        return self.order_history.copy()
