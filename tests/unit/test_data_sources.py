"""Tests for data source interfaces and implementations - TDD RED phase."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterator

import pytest

from firebot.core.models import OHLCV
from firebot.data.sources.base import DataSource
from firebot.data.sources.csv_source import CSVDataSource


class TestDataSourceInterface:
    """Tests for the DataSource abstract base class."""

    def test_data_source_is_abstract(self) -> None:
        """DataSource should not be instantiable directly."""
        with pytest.raises(TypeError):
            DataSource()  # type: ignore

    def test_data_source_requires_get_historical(self) -> None:
        """DataSource subclass must implement get_historical."""

        class IncompleteSource(DataSource):
            def subscribe(self, symbol: str) -> None:
                pass

            def get_symbols(self) -> list[str]:
                return []

        with pytest.raises(TypeError):
            IncompleteSource()  # type: ignore


class TestCSVDataSource:
    """Tests for CSV data source implementation."""

    @pytest.fixture
    def sample_csv_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory with sample CSV data."""
        csv_content = """timestamp,open,high,low,close,volume
2024-01-15 09:30:00,185.50,186.25,185.00,186.00,1000000
2024-01-15 10:30:00,186.00,186.75,185.50,186.50,1200000
2024-01-15 11:30:00,186.50,187.00,186.00,186.75,800000
"""
        csv_file = tmp_path / "AAPL.csv"
        csv_file.write_text(csv_content)
        return tmp_path

    def test_csv_source_loads_data(self, sample_csv_dir: Path) -> None:
        """CSVDataSource should load OHLCV data from CSV files."""
        source = CSVDataSource(data_dir=sample_csv_dir)
        data = list(
            source.get_historical(
                symbol="AAPL",
                start=datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
                end=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
                resolution="1h",
            )
        )
        assert len(data) == 3
        assert all(isinstance(d, OHLCV) for d in data)

    def test_csv_source_returns_correct_ohlcv(self, sample_csv_dir: Path) -> None:
        """CSVDataSource should return correctly parsed OHLCV objects."""
        source = CSVDataSource(data_dir=sample_csv_dir)
        data = list(
            source.get_historical(
                symbol="AAPL",
                start=datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
                end=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
                resolution="1h",
            )
        )
        first_bar = data[0]
        assert first_bar.symbol == "AAPL"
        assert first_bar.open == Decimal("185.50")
        assert first_bar.high == Decimal("186.25")
        assert first_bar.close == Decimal("186.00")
        assert first_bar.volume == Decimal("1000000")

    def test_csv_source_filters_by_date_range(self, sample_csv_dir: Path) -> None:
        """CSVDataSource should filter data by start/end dates."""
        source = CSVDataSource(data_dir=sample_csv_dir)
        data = list(
            source.get_historical(
                symbol="AAPL",
                start=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
                end=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
                resolution="1h",
            )
        )
        # Should only get the 10:30 bar
        assert len(data) == 1
        assert data[0].timestamp.hour == 10

    def test_csv_source_get_symbols(self, sample_csv_dir: Path) -> None:
        """CSVDataSource should list available symbols."""
        source = CSVDataSource(data_dir=sample_csv_dir)
        symbols = source.get_symbols()
        assert "AAPL" in symbols

    def test_csv_source_raises_on_missing_symbol(self, sample_csv_dir: Path) -> None:
        """CSVDataSource should raise FileNotFoundError for missing symbols."""
        source = CSVDataSource(data_dir=sample_csv_dir)
        with pytest.raises(FileNotFoundError):
            list(
                source.get_historical(
                    symbol="UNKNOWN",
                    start=datetime(2024, 1, 15, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 16, tzinfo=timezone.utc),
                    resolution="1h",
                )
            )

    def test_csv_source_returns_iterator(self, sample_csv_dir: Path) -> None:
        """CSVDataSource should return an iterator for memory efficiency."""
        source = CSVDataSource(data_dir=sample_csv_dir)
        result = source.get_historical(
            symbol="AAPL",
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, tzinfo=timezone.utc),
            resolution="1h",
        )
        assert isinstance(result, Iterator)
