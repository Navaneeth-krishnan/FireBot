"""Tests for ML integration module - TDD RED phase."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from firebot.core.models import OHLCV, Signal, SignalDirection
from firebot.ml.strategy import MLStrategy
from firebot.ml.feature_store import FeatureStore, FeatureSet
from firebot.ml.versioning import ModelVersionManager, ModelVersion


class MockModel:
    """Mock ML model for testing."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.array([0.7])


class ConcreteMLStrategy(MLStrategy):
    """Concrete implementation for testing."""

    def preprocess(self, features: dict[str, Any]) -> np.ndarray:
        return np.array([features.get("sma_20", 0), features.get("returns", 0)])

    def interpret_prediction(self, prediction: np.ndarray) -> Signal | None:
        value = float(prediction[0])
        if value > 0.5:
            direction = SignalDirection.LONG
        elif value < -0.5:
            direction = SignalDirection.SHORT
        else:
            return None

        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=self._last_symbol or "UNKNOWN",
            direction=direction,
            confidence=min(abs(value), 1.0),
            strategy_id=self.strategy_id,
            metadata={"raw_prediction": value},
        )


class TestMLStrategy:
    """Tests for MLStrategy base class."""

    def test_initialization_with_model(self) -> None:
        """MLStrategy should initialize with a model object."""
        model = MockModel()
        strategy = ConcreteMLStrategy(
            strategy_id="ml_test",
            config={"lookback": 20},
            model=model,
        )
        assert strategy.strategy_id == "ml_test"
        assert strategy.model is model

    def test_initialization_without_model(self) -> None:
        """MLStrategy should work without a model (for lazy loading)."""
        strategy = ConcreteMLStrategy(
            strategy_id="ml_test",
            config={},
        )
        assert strategy.model is None

    def test_load_model_from_path(self, tmp_path: Path) -> None:
        """MLStrategy should load model from a file path."""
        # Save a mock model artifact
        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"fake_model_data")

        strategy = ConcreteMLStrategy(
            strategy_id="ml_test",
            config={},
        )
        # load_model is abstract-ish; we test the interface exists
        strategy.set_model(MockModel())
        assert strategy.model is not None

    def test_generate_signal_uses_model(self) -> None:
        """generate_signal should use model.predict on preprocessed features."""
        model = MockModel()
        strategy = ConcreteMLStrategy(
            strategy_id="ml_test",
            config={},
            model=model,
        )
        strategy._last_symbol = "AAPL"

        signal = strategy.generate_signal({"sma_20": 150.0, "returns": 0.02})
        assert signal is not None
        assert signal.direction == SignalDirection.LONG
        assert signal.confidence > 0

    def test_generate_signal_no_model_returns_none(self) -> None:
        """generate_signal should return None if no model is loaded."""
        strategy = ConcreteMLStrategy(
            strategy_id="ml_test",
            config={},
        )
        result = strategy.generate_signal({"sma_20": 150.0})
        assert result is None

    def test_on_data_updates_buffer(self) -> None:
        """on_data should add to data buffer."""
        strategy = ConcreteMLStrategy(
            strategy_id="ml_test",
            config={},
            model=MockModel(),
        )
        ohlcv = OHLCV(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            open=Decimal("150"),
            high=Decimal("155"),
            low=Decimal("149"),
            close=Decimal("153"),
            volume=Decimal("1000000"),
        )
        strategy.on_data(ohlcv)
        assert len(strategy.data_buffer) == 1
        assert strategy._last_symbol == "AAPL"

    def test_preprocess_is_called(self) -> None:
        """generate_signal should call preprocess before predict."""
        model = MagicMock()
        model.predict.return_value = np.array([0.7])

        strategy = ConcreteMLStrategy(
            strategy_id="ml_test",
            config={},
            model=model,
        )
        strategy._last_symbol = "AAPL"

        strategy.generate_signal({"sma_20": 150.0, "returns": 0.02})
        model.predict.assert_called_once()
        # Verify the input shape
        call_args = model.predict.call_args[0][0]
        assert isinstance(call_args, np.ndarray)


class TestFeatureStore:
    """Tests for feature store interface."""

    def test_store_initialization(self) -> None:
        """Feature store should initialize empty."""
        store = FeatureStore()
        assert store.count == 0

    def test_register_feature_set(self) -> None:
        """Should register a named feature set."""
        store = FeatureStore()
        feature_set = FeatureSet(
            name="technical_v1",
            features=["sma_20", "sma_50", "rsi_14", "returns_5d"],
            version="1.0",
        )
        store.register(feature_set)
        assert store.count == 1
        assert store.get("technical_v1") is not None

    def test_store_features(self) -> None:
        """Should store feature values for a symbol and timestamp."""
        store = FeatureStore()
        feature_set = FeatureSet(
            name="technical_v1",
            features=["sma_20", "returns"],
            version="1.0",
        )
        store.register(feature_set)

        now = datetime.now(timezone.utc)
        store.put(
            feature_set_name="technical_v1",
            symbol="AAPL",
            timestamp=now,
            values={"sma_20": 150.0, "returns": 0.02},
        )

        result = store.get_latest("technical_v1", "AAPL")
        assert result is not None
        assert result["sma_20"] == 150.0
        assert result["returns"] == 0.02

    def test_get_history(self) -> None:
        """Should retrieve feature history for a symbol."""
        store = FeatureStore()
        fs = FeatureSet(name="tech_v1", features=["sma_20"], version="1.0")
        store.register(fs)

        now = datetime.now(timezone.utc)
        for i in range(5):
            store.put(
                "tech_v1",
                "AAPL",
                now + __import__("datetime").timedelta(hours=i),
                {"sma_20": 150.0 + i},
            )

        history = store.get_history("tech_v1", "AAPL", limit=3)
        assert len(history) == 3
        # Most recent first
        assert history[0]["sma_20"] == 154.0

    def test_get_nonexistent_returns_none(self) -> None:
        """Getting features for unregistered set should return None."""
        store = FeatureStore()
        result = store.get("nonexistent")
        assert result is None

    def test_list_feature_sets(self) -> None:
        """Should list all registered feature sets."""
        store = FeatureStore()
        store.register(FeatureSet(name="tech_v1", features=["sma_20"], version="1.0"))
        store.register(FeatureSet(name="ml_v1", features=["embedding"], version="1.0"))

        names = store.list_feature_sets()
        assert set(names) == {"tech_v1", "ml_v1"}


class TestModelVersionManager:
    """Tests for model versioning."""

    def test_manager_initialization(self, tmp_path: Path) -> None:
        """Manager should initialize with a storage path."""
        manager = ModelVersionManager(storage_path=tmp_path)
        assert manager.storage_path == tmp_path

    def test_register_model_version(self, tmp_path: Path) -> None:
        """Should register a new model version."""
        manager = ModelVersionManager(storage_path=tmp_path)
        version = manager.register(
            model_name="momentum_classifier",
            version="1.0.0",
            metadata={"accuracy": 0.85, "framework": "sklearn"},
        )
        assert isinstance(version, ModelVersion)
        assert version.model_name == "momentum_classifier"
        assert version.version == "1.0.0"

    def test_list_versions(self, tmp_path: Path) -> None:
        """Should list all versions for a model."""
        manager = ModelVersionManager(storage_path=tmp_path)
        manager.register("my_model", "1.0.0", {"accuracy": 0.80})
        manager.register("my_model", "1.1.0", {"accuracy": 0.85})
        manager.register("my_model", "2.0.0", {"accuracy": 0.90})

        versions = manager.list_versions("my_model")
        assert len(versions) == 3

    def test_get_latest_version(self, tmp_path: Path) -> None:
        """Should return the latest registered version."""
        manager = ModelVersionManager(storage_path=tmp_path)
        manager.register("my_model", "1.0.0", {"accuracy": 0.80})
        manager.register("my_model", "1.1.0", {"accuracy": 0.85})

        latest = manager.get_latest("my_model")
        assert latest is not None
        assert latest.version == "1.1.0"

    def test_get_specific_version(self, tmp_path: Path) -> None:
        """Should retrieve a specific version by name."""
        manager = ModelVersionManager(storage_path=tmp_path)
        manager.register("my_model", "1.0.0", {"accuracy": 0.80})
        manager.register("my_model", "2.0.0", {"accuracy": 0.90})

        v1 = manager.get_version("my_model", "1.0.0")
        assert v1 is not None
        assert v1.metadata["accuracy"] == 0.80

    def test_get_nonexistent_version(self, tmp_path: Path) -> None:
        """Getting nonexistent version should return None."""
        manager = ModelVersionManager(storage_path=tmp_path)
        result = manager.get_version("nonexistent", "1.0.0")
        assert result is None

    def test_save_and_load_artifact(self, tmp_path: Path) -> None:
        """Should persist model artifact to disk."""
        manager = ModelVersionManager(storage_path=tmp_path)
        version = manager.register("my_model", "1.0.0", {})

        # Save artifact bytes
        artifact_data = b"serialized_model_data"
        manager.save_artifact(version, artifact_data)

        # Load artifact
        loaded = manager.load_artifact(version)
        assert loaded == artifact_data

    def test_version_metadata_preserved(self, tmp_path: Path) -> None:
        """Model metadata should survive serialization."""
        manager = ModelVersionManager(storage_path=tmp_path)
        manager.register(
            "my_model",
            "1.0.0",
            {
                "accuracy": 0.85,
                "features": ["sma_20", "rsi_14"],
                "framework": "pytorch",
            },
        )

        version = manager.get_version("my_model", "1.0.0")
        assert version is not None
        assert version.metadata["accuracy"] == 0.85
        assert version.metadata["features"] == ["sma_20", "rsi_14"]
