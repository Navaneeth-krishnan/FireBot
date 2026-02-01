"""Transformer-based trading strategy example.

Demonstrates how to build an ML strategy using PyTorch Transformer
architecture for time-series prediction. This serves as a reference
implementation for building custom ML strategies.

Note: Requires the 'ml' optional dependency group (torch).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from firebot.core.models import OHLCV, Signal, SignalDirection
from firebot.ml.strategy import MLStrategy

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _SimpleTransformerModel:
    """Lightweight wrapper around a PyTorch Transformer for prediction.

    Wraps a nn.TransformerEncoder to produce a single directional
    prediction from a sequence of feature vectors.

    Architecture:
        Input -> Linear projection -> Positional encoding ->
        TransformerEncoder -> Mean pooling -> Linear head -> Tanh

    Output range: [-1.0, 1.0] where:
        > 0.0: bullish signal
        < 0.0: bearish signal
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 20,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("torch is required. Install with: uv sync --extra ml")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
        self.activation = nn.Tanh()

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on a feature array.

        Args:
            features: numpy array of shape (seq_len, input_dim) or (input_dim,)

        Returns:
            numpy array with single prediction in [-1, 1]
        """
        if not HAS_TORCH:
            raise ImportError("torch is required")

        with torch.no_grad():
            x = torch.FloatTensor(features)

            # Handle single-step input: expand to (1, 1, input_dim)
            if x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension

            projected = self.input_projection(x)
            encoded = self.encoder(projected)
            pooled = encoded.mean(dim=1)  # Mean pooling over sequence
            output = self.activation(self.head(pooled))

        return output.numpy().flatten()


class TransformerStrategy(MLStrategy):
    """Reference Transformer-based ML trading strategy.

    Uses a small Transformer encoder to predict directional signals
    from a rolling window of technical features.

    Configuration:
        input_dim: Number of input features (default: 5)
        seq_len: Lookback window size (default: 20)
        d_model: Transformer hidden dimension (default: 32)
        nhead: Number of attention heads (default: 4)
        num_layers: Transformer layers (default: 2)
        long_threshold: Prediction threshold for LONG (default: 0.3)
        short_threshold: Prediction threshold for SHORT (default: -0.3)
        feature_keys: List of feature names to use from features dict

    Example:
        strategy = TransformerStrategy(
            strategy_id="transformer_1",
            config={
                "input_dim": 5,
                "seq_len": 20,
                "feature_keys": ["sma_20", "sma_50", "returns", "volatility", "volume_norm"],
            },
        )
    """

    def __init__(
        self,
        strategy_id: str,
        config: dict[str, Any],
        model: Any | None = None,
    ) -> None:
        if model is None and HAS_TORCH:
            model = _SimpleTransformerModel(
                input_dim=config.get("input_dim", 5),
                d_model=config.get("d_model", 32),
                nhead=config.get("nhead", 4),
                num_layers=config.get("num_layers", 2),
                seq_len=config.get("seq_len", 20),
            )

        super().__init__(strategy_id, config, model)

        self.seq_len: int = config.get("seq_len", 20)
        self.long_threshold: float = config.get("long_threshold", 0.3)
        self.short_threshold: float = config.get("short_threshold", -0.3)
        self.feature_keys: list[str] = config.get(
            "feature_keys",
            ["sma_20", "sma_50", "returns", "volatility", "volume"],
        )
        self._feature_history: list[np.ndarray] = []

    def on_data(self, data: OHLCV) -> None:
        """Process new market data and update feature history."""
        super().on_data(data)

    def preprocess(self, features: dict[str, Any]) -> np.ndarray:
        """Convert features dict to numpy array and maintain rolling window.

        Args:
            features: Feature dictionary with keys matching feature_keys

        Returns:
            Numpy array of shape (seq_len, input_dim) or (1, input_dim)
        """
        feature_vector = np.array(
            [float(features.get(k, 0.0)) for k in self.feature_keys],
            dtype=np.float32,
        )
        self._feature_history.append(feature_vector)

        # Trim to seq_len
        if len(self._feature_history) > self.seq_len:
            self._feature_history = self._feature_history[-self.seq_len:]

        return np.array(self._feature_history, dtype=np.float32)

    def interpret_prediction(self, prediction: np.ndarray) -> Signal | None:
        """Convert model prediction to trading signal.

        Args:
            prediction: Model output in [-1, 1] range

        Returns:
            Signal with direction based on thresholds, or None
        """
        value = float(prediction[0])

        if value > self.long_threshold:
            direction = SignalDirection.LONG
        elif value < self.short_threshold:
            direction = SignalDirection.SHORT
        else:
            return None

        timestamp = (
            self.data_buffer[-1].timestamp
            if self.data_buffer
            else datetime.now(timezone.utc)
        )
        symbol = self._last_symbol or "UNKNOWN"

        return Signal(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            confidence=min(abs(value), 1.0),
            strategy_id=self.strategy_id,
            metadata={
                "raw_prediction": value,
                "seq_len_used": len(self._feature_history),
            },
        )

    def reset(self) -> None:
        """Reset strategy state including feature history."""
        super().reset()
        self._feature_history.clear()
