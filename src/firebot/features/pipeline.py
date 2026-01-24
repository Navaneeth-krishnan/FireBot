"""Abstract base class for feature pipelines."""

from abc import ABC, abstractmethod
from typing import Any

from firebot.core.models import OHLCV


class FeaturePipeline(ABC):
    """Abstract base class for feature engineering pipelines.

    Feature pipelines transform raw OHLCV data into features
    suitable for trading strategies and ML models.
    """

    @abstractmethod
    def transform(self, data: list[OHLCV]) -> dict[str, Any]:
        """Transform raw OHLCV data into features.

        Args:
            data: List of OHLCV bars in chronological order

        Returns:
            Dictionary mapping feature names to their computed values

        Raises:
            ValueError: If the data is empty or invalid
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Get list of feature names produced by this pipeline.

        Returns:
            List of feature names that will be in the transform output
        """
        pass
