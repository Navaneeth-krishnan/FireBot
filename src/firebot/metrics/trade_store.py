"""Time-series trade history storage."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class TradeEvent:
    """Immutable record of a trade event."""

    timestamp: datetime
    strategy_id: str
    symbol: str
    side: str
    quantity: str  # String to preserve Decimal precision
    entry_price: str
    exit_price: str
    pnl: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TradeStore:
    """Time-series storage for trade history.

    Stores trade events in chronological order with support for
    querying by strategy, symbol, and time range. Uses append-only
    JSON lines format for durability.

    Example:
        store = TradeStore()
        store.record(
            strategy_id="momentum_1",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            entry_price=Decimal("150"),
            exit_price=Decimal("160"),
            pnl=Decimal("1000"),
        )
        trades = store.query(strategy_id="momentum_1")
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        """Initialize trade store.

        Args:
            persist_path: Optional file path for persistent storage (JSONL)
        """
        self._trades: list[TradeEvent] = []
        self._persist_path = persist_path

        # Load existing trades from disk if path exists
        if persist_path and persist_path.exists():
            self._load_from_disk()

    def record(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        exit_price: Decimal,
        pnl: Decimal,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TradeEvent:
        """Record a completed trade.

        Args:
            strategy_id: Strategy that executed the trade
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Trade quantity
            entry_price: Entry price
            exit_price: Exit price
            pnl: Realized PnL
            timestamp: Trade timestamp (defaults to now)
            metadata: Optional additional data

        Returns:
            The created TradeEvent
        """
        event = TradeEvent(
            timestamp=timestamp or datetime.now(timezone.utc),
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            quantity=str(quantity),
            entry_price=str(entry_price),
            exit_price=str(exit_price),
            pnl=str(pnl),
            metadata=metadata or {},
        )

        self._trades.append(event)

        if self._persist_path:
            self._append_to_disk(event)

        return event

    def query(
        self,
        strategy_id: str | None = None,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[TradeEvent]:
        """Query trade history with filters.

        Args:
            strategy_id: Filter by strategy
            symbol: Filter by symbol
            start: Filter trades after this time
            end: Filter trades before this time

        Returns:
            List of matching TradeEvents
        """
        results = self._trades

        if strategy_id is not None:
            results = [t for t in results if t.strategy_id == strategy_id]

        if symbol is not None:
            results = [t for t in results if t.symbol == symbol]

        if start is not None:
            results = [t for t in results if t.timestamp >= start]

        if end is not None:
            results = [t for t in results if t.timestamp <= end]

        return results

    def get_strategies(self) -> list[str]:
        """Get list of unique strategy IDs.

        Returns:
            List of strategy IDs
        """
        return list({t.strategy_id for t in self._trades})

    def get_symbols(self) -> list[str]:
        """Get list of unique symbols traded.

        Returns:
            List of symbols
        """
        return list({t.symbol for t in self._trades})

    @property
    def count(self) -> int:
        """Total number of trades stored."""
        return len(self._trades)

    def _append_to_disk(self, event: TradeEvent) -> None:
        """Append a trade event to the JSONL file."""
        if self._persist_path is None:
            return

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        record = asdict(event)
        record["timestamp"] = event.timestamp.isoformat()

        with open(self._persist_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _load_from_disk(self) -> None:
        """Load trades from JSONL file."""
        if self._persist_path is None or not self._persist_path.exists():
            return

        with open(self._persist_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["timestamp"] = datetime.fromisoformat(record["timestamp"])
                self._trades.append(TradeEvent(**record))
