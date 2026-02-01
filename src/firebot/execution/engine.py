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
    status: str  # "filled", "rejected", "pending"
    fill_price: Decimal | None
    fill_quantity: Decimal | None
    timestamp: datetime
    message: str = ""


class PaperTradingEngine:
    """Simulates order execution for paper trading.

    Supports:
    - Instant fill at current price (market orders)
    - Configurable slippage
    - Stop-loss and take-profit conditional orders
    - Pending order tracking and price-triggered execution

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
        self._pending_orders: list[Order] = []

    def submit_order(
        self,
        order: Order,
        current_price: Decimal,
    ) -> FillResult:
        """Submit an order for execution.

        Market orders are filled instantly. Stop-loss and take-profit
        orders are checked against the current price and either filled
        or queued as pending.

        Args:
            order: Order to execute
            current_price: Current market price for the symbol

        Returns:
            FillResult with execution details
        """
        if order.order_type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT):
            return self._handle_conditional_order(order, current_price)

        if self.fill_model == "instant":
            return self._instant_fill(order, current_price)
        else:
            raise ValueError(f"Unknown fill model: {self.fill_model}")

    def _handle_conditional_order(
        self, order: Order, current_price: Decimal
    ) -> FillResult:
        """Handle stop-loss and take-profit orders.

        Logic:
        - Sell stop-loss: triggers when price <= stop price
        - Buy stop-loss: triggers when price >= stop price (short cover)
        - Sell take-profit: triggers when price >= target
        - Buy take-profit: triggers when price <= target

        Args:
            order: Conditional order
            current_price: Current market price

        Returns:
            FillResult (filled if triggered, pending otherwise)
        """
        triggered = self._is_conditional_triggered(order, current_price)

        if triggered:
            return self._instant_fill(order, current_price)

        # Not triggered - add to pending
        self._pending_orders.append(order)
        result = FillResult(
            order_id=order.id,
            status="pending",
            fill_price=None,
            fill_quantity=None,
            timestamp=datetime.now(timezone.utc),
            message=f"Conditional order pending: {order.order_type.value}",
        )
        return result

    def _is_conditional_triggered(
        self, order: Order, current_price: Decimal
    ) -> bool:
        """Check if a conditional order should trigger.

        Args:
            order: The conditional order
            current_price: Current market price

        Returns:
            True if the order should be filled
        """
        stop_price = order.price
        if stop_price is None:
            return False

        if order.order_type == OrderType.STOP_LOSS:
            if order.side == OrderSide.SELL:
                # Sell stop: trigger when price falls to or below stop
                return current_price <= stop_price
            else:
                # Buy stop (short cover): trigger when price rises to or above stop
                return current_price >= stop_price

        if order.order_type == OrderType.TAKE_PROFIT:
            if order.side == OrderSide.SELL:
                # Sell TP: trigger when price rises to or above target
                return current_price >= stop_price
            else:
                # Buy TP: trigger when price falls to or below target
                return current_price <= stop_price

        return False

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

    def get_pending_orders(self) -> list[Order]:
        """Get all pending conditional orders.

        Returns:
            List of pending orders
        """
        return list(self._pending_orders)

    def check_pending_orders(
        self, current_prices: dict[str, Decimal]
    ) -> list[FillResult]:
        """Check pending orders against new prices and trigger if conditions met.

        Args:
            current_prices: Dict mapping symbol to current price

        Returns:
            List of FillResults for newly triggered orders
        """
        triggered_results = []
        still_pending = []

        for order in self._pending_orders:
            price = current_prices.get(order.symbol)
            if price is None:
                still_pending.append(order)
                continue

            if self._is_conditional_triggered(order, price):
                result = self._instant_fill(order, price)
                triggered_results.append(result)
            else:
                still_pending.append(order)

        self._pending_orders = still_pending
        return triggered_results

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

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancelled, False if not found
        """
        for i, order in enumerate(self._pending_orders):
            if order.id == order_id:
                self._pending_orders.pop(i)
                return True
        return False

    def get_order_history(self) -> list[tuple[Order, FillResult]]:
        """Get complete order history.

        Returns:
            List of (order, result) tuples
        """
        return list(self.order_history)
