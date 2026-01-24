"""Tests for feature engineering pipeline - TDD RED phase."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from firebot.core.models import OHLCV
from firebot.features.pipeline import FeaturePipeline
from firebot.features.technical import TechnicalFeatures


class TestFeaturePipelineInterface:
    """Tests for the FeaturePipeline abstract base class."""

    def test_feature_pipeline_is_abstract(self) -> None:
        """FeaturePipeline should not be instantiable directly."""
        with pytest.raises(TypeError):
            FeaturePipeline()  # type: ignore

    def test_feature_pipeline_requires_transform(self) -> None:
        """FeaturePipeline subclass must implement transform."""

        class IncompletePipeline(FeaturePipeline):
            def get_feature_names(self) -> list[str]:
                return []

        with pytest.raises(TypeError):
            IncompletePipeline()  # type: ignore


class TestTechnicalFeatures:
    """Tests for technical indicator feature pipeline."""

    @pytest.fixture
    def sample_ohlcv_data(self) -> list[OHLCV]:
        """Create sample OHLCV data for testing."""
        base_time = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        prices = [
            (100, 102, 99, 101, 1000),
            (101, 103, 100, 102, 1100),
            (102, 104, 101, 103, 1200),
            (103, 105, 102, 104, 1000),
            (104, 106, 103, 105, 900),
            (105, 107, 104, 106, 1100),
            (106, 108, 105, 107, 1300),
            (107, 109, 106, 108, 1200),
            (108, 110, 107, 109, 1000),
            (109, 111, 108, 110, 1100),
        ]
        return [
            OHLCV(
                timestamp=base_time.replace(hour=9 + i),
                symbol="TEST",
                open=Decimal(str(p[0])),
                high=Decimal(str(p[1])),
                low=Decimal(str(p[2])),
                close=Decimal(str(p[3])),
                volume=Decimal(str(p[4])),
                resolution="1h",
            )
            for i, p in enumerate(prices)
        ]

    def test_technical_features_transform_returns_dict(
        self, sample_ohlcv_data: list[OHLCV]
    ) -> None:
        """TechnicalFeatures.transform() should return a dictionary."""
        pipeline = TechnicalFeatures(sma_periods=[3, 5])
        features = pipeline.transform(sample_ohlcv_data)
        assert isinstance(features, dict)

    def test_technical_features_calculates_sma(
        self, sample_ohlcv_data: list[OHLCV]
    ) -> None:
        """TechnicalFeatures should calculate simple moving averages."""
        pipeline = TechnicalFeatures(sma_periods=[3])
        features = pipeline.transform(sample_ohlcv_data)
        assert "sma_3" in features
        # SMA of last 3 closes: (108 + 109 + 110) / 3 = 109
        assert features["sma_3"] == pytest.approx(109.0, rel=0.01)

    def test_technical_features_calculates_returns(
        self, sample_ohlcv_data: list[OHLCV]
    ) -> None:
        """TechnicalFeatures should calculate returns."""
        pipeline = TechnicalFeatures()
        features = pipeline.transform(sample_ohlcv_data)
        assert "returns" in features
        # Return = (110 - 109) / 109 = 0.00917...
        assert features["returns"] == pytest.approx(0.00917, rel=0.01)

    def test_technical_features_calculates_volatility(
        self, sample_ohlcv_data: list[OHLCV]
    ) -> None:
        """TechnicalFeatures should calculate volatility (std of returns)."""
        pipeline = TechnicalFeatures(volatility_period=5)
        features = pipeline.transform(sample_ohlcv_data)
        assert "volatility" in features
        assert features["volatility"] >= 0

    def test_technical_features_get_feature_names(self) -> None:
        """TechnicalFeatures should return list of feature names."""
        pipeline = TechnicalFeatures(sma_periods=[3, 5, 10])
        names = pipeline.get_feature_names()
        assert "sma_3" in names
        assert "sma_5" in names
        assert "sma_10" in names
        assert "returns" in names
        assert "volatility" in names

    def test_technical_features_handles_insufficient_data(self) -> None:
        """TechnicalFeatures should handle insufficient data gracefully."""
        pipeline = TechnicalFeatures(sma_periods=[20])  # Need 20 bars for SMA
        # Only provide 5 bars
        short_data = [
            OHLCV(
                timestamp=datetime(2024, 1, 15, 9 + i, tzinfo=timezone.utc),
                symbol="TEST",
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=Decimal("1000"),
                resolution="1h",
            )
            for i in range(5)
        ]
        features = pipeline.transform(short_data)
        # Should return NaN for SMA when insufficient data
        assert features["sma_20"] is None or features["sma_20"] != features["sma_20"]

    def test_technical_features_empty_data_raises(self) -> None:
        """TechnicalFeatures should raise ValueError for empty data."""
        pipeline = TechnicalFeatures()
        with pytest.raises(ValueError, match="empty"):
            pipeline.transform([])
