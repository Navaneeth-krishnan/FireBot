"""CSV file data source implementation."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterator

import pandas as pd

from firebot.core.models import OHLCV
from firebot.data.sources.base import DataSource


class CSVDataSource(DataSource):
    """Data source that reads OHLCV data from CSV files.

    Expected CSV format:
        timestamp,open,high,low,close,volume
        2024-01-15 09:30:00,185.50,186.25,185.00,186.00,1000000

    File naming convention: {symbol}.csv (e.g., AAPL.csv)
    """

    def __init__(self, data_dir: Path | str) -> None:
        """Initialize CSV data source.

        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

    def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        resolution: str = "1h",
    ) -> Iterator[OHLCV]:
        """Fetch historical OHLCV data from CSV file.

        Args:
            symbol: The ticker symbol (corresponds to filename)
            start: Start datetime for the data range
            end: End datetime for the data range
            resolution: Data resolution (stored in OHLCV but not used for filtering)

        Yields:
            OHLCV objects in chronological order

        Raises:
            FileNotFoundError: If the CSV file for the symbol doesn't exist
        """
        csv_path = self.data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No data file found for symbol: {symbol}")

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])

        # Ensure timestamps are timezone-aware
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(timezone.utc)

        # Convert start/end to timezone-aware if needed
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        # Filter by date range
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df_filtered = df[mask].sort_values("timestamp")

        for _, row in df_filtered.iterrows():
            yield OHLCV(
                timestamp=row["timestamp"].to_pydatetime(),
                symbol=symbol,
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=Decimal(str(row["volume"])),
                resolution=resolution,
            )

    def get_symbols(self) -> list[str]:
        """Get list of available symbols from CSV filenames.

        Returns:
            List of symbols (CSV filenames without extension)
        """
        return [f.stem for f in self.data_dir.glob("*.csv")]

    def subscribe(self, symbol: str) -> None:
        """Subscribe to live data (not supported for CSV source)."""
        raise NotImplementedError("CSV data source does not support live data")
