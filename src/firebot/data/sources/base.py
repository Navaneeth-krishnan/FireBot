"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator

from firebot.core.models import OHLCV


class DataSource(ABC):
    """Abstract base class for market data sources.

    All data sources must implement this interface to be compatible
    with the FireBot trading system.
    """

    @abstractmethod
    def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        resolution: str = "1h",
    ) -> Iterator[OHLCV]:
        """Fetch historical OHLCV data for a symbol.

        Args:
            symbol: The ticker symbol (e.g., "AAPL", "GOOGL")
            start: Start datetime for the data range
            end: End datetime for the data range
            resolution: Data resolution (e.g., "1m", "5m", "1h", "1d")

        Yields:
            OHLCV objects in chronological order

        Raises:
            FileNotFoundError: If the symbol data is not available
        """
        pass

    def subscribe(self, symbol: str) -> None:
        """Subscribe to live data for a symbol.

        This is optional and may not be implemented by all sources.
        Live data is a future feature.

        Args:
            symbol: The ticker symbol to subscribe to
        """
        raise NotImplementedError("Live data subscription not supported")

    @abstractmethod
    def get_symbols(self) -> list[str]:
        """Get list of available symbols.

        Returns:
            List of ticker symbols available from this source
        """
        pass
