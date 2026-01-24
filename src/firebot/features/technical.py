"""Technical indicator feature pipeline."""

import math
from typing import Any

from firebot.core.models import OHLCV
from firebot.features.pipeline import FeaturePipeline


class TechnicalFeatures(FeaturePipeline):
    """Feature pipeline that calculates technical indicators.

    Calculates common technical analysis features from OHLCV data:
    - Simple Moving Averages (SMA) at configurable periods
    - Returns (percentage change)
    - Volatility (standard deviation of returns)
    """

    def __init__(
        self,
        sma_periods: list[int] | None = None,
        volatility_period: int = 20,
    ) -> None:
        """Initialize technical features pipeline.

        Args:
            sma_periods: List of periods for SMA calculation (default: [5, 10, 20])
            volatility_period: Period for volatility calculation (default: 20)
        """
        self.sma_periods = sma_periods or [5, 10, 20]
        self.volatility_period = volatility_period

    def transform(self, data: list[OHLCV]) -> dict[str, Any]:
        """Transform OHLCV data into technical features.

        Args:
            data: List of OHLCV bars in chronological order

        Returns:
            Dictionary with calculated features

        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError("Cannot transform empty data")

        closes = [float(bar.close) for bar in data]
        features: dict[str, Any] = {}

        # Calculate SMAs
        for period in self.sma_periods:
            sma_key = f"sma_{period}"
            if len(closes) >= period:
                features[sma_key] = sum(closes[-period:]) / period
            else:
                features[sma_key] = float("nan")

        # Calculate returns
        if len(closes) >= 2:
            features["returns"] = (closes[-1] - closes[-2]) / closes[-2]
        else:
            features["returns"] = 0.0

        # Calculate volatility (std of returns)
        if len(closes) >= self.volatility_period:
            returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
            ]
            recent_returns = returns[-self.volatility_period :]
            mean_return = sum(recent_returns) / len(recent_returns)
            variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(
                recent_returns
            )
            features["volatility"] = math.sqrt(variance)
        else:
            # Calculate with available data
            if len(closes) >= 2:
                returns = [
                    (closes[i] - closes[i - 1]) / closes[i - 1]
                    for i in range(1, len(closes))
                ]
                if returns:
                    mean_return = sum(returns) / len(returns)
                    variance = sum((r - mean_return) ** 2 for r in returns) / len(
                        returns
                    )
                    features["volatility"] = math.sqrt(variance)
                else:
                    features["volatility"] = 0.0
            else:
                features["volatility"] = 0.0

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names.

        Returns:
            List of feature names produced by this pipeline
        """
        names = [f"sma_{period}" for period in self.sma_periods]
        names.extend(["returns", "volatility"])
        return names
