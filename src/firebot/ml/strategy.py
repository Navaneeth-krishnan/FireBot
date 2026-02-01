"""ML Strategy base class for model-driven trading strategies."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from firebot.core.models import OHLCV, Signal
from firebot.strategies.base import Strategy


class MLStrategy(Strategy):
    """Base class for ML-driven trading strategies.

    Extends the Strategy base class with model loading, preprocessing,
    and prediction interpretation. Subclasses must implement:
    - preprocess(): Convert feature dict to model input (numpy array)
    - interpret_prediction(): Convert model output to a Signal

    The generate_signal() method orchestrates the full pipeline:
    features -> preprocess -> model.predict -> interpret_prediction

    Example:
        class MyMLStrategy(MLStrategy):
            def preprocess(self, features):
                return np.array([features["sma_20"], features["rsi"]])

            def interpret_prediction(self, prediction):
                if prediction[0] > 0.5:
                    return Signal(direction=SignalDirection.LONG, ...)
                return None
    """

    def __init__(
        self,
        strategy_id: str,
        config: dict[str, Any],
        model: Any | None = None,
    ) -> None:
        """Initialize ML strategy.

        Args:
            strategy_id: Unique strategy identifier
            config: Strategy configuration parameters
            model: Optional pre-loaded model object with a predict() method
        """
        super().__init__(strategy_id, config)
        self.model = model
        self._last_symbol: str = ""

    def set_model(self, model: Any) -> None:
        """Set or replace the model.

        Args:
            model: Model object with a predict() method
        """
        self.model = model

    def on_data(self, data: OHLCV) -> None:
        """Process new market data.

        Args:
            data: New OHLCV bar
        """
        self.data_buffer.append(data)
        self._last_symbol = data.symbol

    def generate_signal(self, features: dict[str, Any]) -> Signal | None:
        """Generate a trading signal using the ML model.

        Pipeline: features -> preprocess -> predict -> interpret

        Args:
            features: Dictionary of computed features

        Returns:
            Signal if model produces actionable prediction, None otherwise
        """
        if self.model is None:
            return None

        preprocessed = self.preprocess(features)
        prediction = self.model.predict(preprocessed)
        return self.interpret_prediction(prediction)

    @abstractmethod
    def preprocess(self, features: dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to model input.

        Args:
            features: Raw feature dictionary from FeaturePipeline

        Returns:
            Numpy array suitable for model.predict()
        """
        ...

    @abstractmethod
    def interpret_prediction(self, prediction: np.ndarray) -> Signal | None:
        """Convert model output to a trading Signal.

        Args:
            prediction: Raw model output array

        Returns:
            Signal if prediction is actionable, None otherwise
        """
        ...
