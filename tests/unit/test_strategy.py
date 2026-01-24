"""Tests for Strategy base class and registry - TDD RED phase."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from firebot.core.models import OHLCV, Signal, SignalDirection
from firebot.strategies.base import Strategy
from firebot.strategies.registry import StrategyRegistry
from firebot.strategies.momentum import MomentumStrategy


class TestStrategyInterface:
    """Tests for the Strategy abstract base class."""

    def test_strategy_is_abstract(self) -> None:
        """Strategy should not be instantiable directly."""
        with pytest.raises(TypeError):
            Strategy(strategy_id="test", config={})  # type: ignore

    def test_strategy_requires_on_data(self) -> None:
        """Strategy subclass must implement on_data."""

        class IncompleteStrategy(Strategy):
            def generate_signal(self, features: dict) -> Signal | None:
                return None

        with pytest.raises(TypeError):
            IncompleteStrategy(strategy_id="test", config={})  # type: ignore

    def test_strategy_requires_generate_signal(self) -> None:
        """Strategy subclass must implement generate_signal."""

        class IncompleteStrategy(Strategy):
            def on_data(self, data: OHLCV) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteStrategy(strategy_id="test", config={})  # type: ignore


class TestStrategyRegistry:
    """Tests for the Strategy plugin registry."""

    def test_register_strategy_class(self) -> None:
        """Registry should allow registering strategy classes."""
        registry = StrategyRegistry()

        @registry.register("momentum")
        class TestMomentum(Strategy):
            def on_data(self, data: OHLCV) -> None:
                pass

            def generate_signal(self, features: dict) -> Signal | None:
                return None

        assert "momentum" in registry.list_strategies()

    def test_get_registered_strategy(self) -> None:
        """Registry should return registered strategy class."""
        registry = StrategyRegistry()

        @registry.register("test_strat")
        class TestStrategy(Strategy):
            def on_data(self, data: OHLCV) -> None:
                pass

            def generate_signal(self, features: dict) -> Signal | None:
                return None

        strategy_cls = registry.get("test_strat")
        assert strategy_cls is TestStrategy

    def test_create_strategy_instance(self) -> None:
        """Registry should create configured strategy instances."""
        registry = StrategyRegistry()

        @registry.register("configurable")
        class ConfigurableStrategy(Strategy):
            def on_data(self, data: OHLCV) -> None:
                pass

            def generate_signal(self, features: dict) -> Signal | None:
                return None

        instance = registry.create(
            "configurable",
            strategy_id="my_strat",
            config={"param1": 10},
        )
        assert instance.strategy_id == "my_strat"
        assert instance.config["param1"] == 10

    def test_get_unregistered_raises(self) -> None:
        """Registry should raise KeyError for unregistered strategies."""
        registry = StrategyRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_duplicate_registration_raises(self) -> None:
        """Registry should raise ValueError for duplicate names."""
        registry = StrategyRegistry()

        @registry.register("duplicate")
        class First(Strategy):
            def on_data(self, data: OHLCV) -> None:
                pass

            def generate_signal(self, features: dict) -> Signal | None:
                return None

        with pytest.raises(ValueError, match="already registered"):

            @registry.register("duplicate")
            class Second(Strategy):
                def on_data(self, data: OHLCV) -> None:
                    pass

                def generate_signal(self, features: dict) -> Signal | None:
                    return None


class TestMomentumStrategy:
    """Tests for the MomentumStrategy reference implementation."""

    @pytest.fixture
    def sample_ohlcv_data(self) -> list[OHLCV]:
        """Create sample OHLCV data with upward momentum."""
        base_time = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        # Prices trending up
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 110]
        return [
            OHLCV(
                timestamp=base_time.replace(hour=9 + i),
                symbol="TEST",
                open=Decimal(str(p - 0.5)),
                high=Decimal(str(p + 0.5)),
                low=Decimal(str(p - 1)),
                close=Decimal(str(p)),
                volume=Decimal("1000"),
                resolution="1h",
            )
            for i, p in enumerate(prices)
        ]

    def test_momentum_strategy_initialization(self) -> None:
        """MomentumStrategy should initialize with config."""
        strategy = MomentumStrategy(
            strategy_id="mom_v1",
            config={"lookback_window": 5, "threshold": 0.02},
        )
        assert strategy.strategy_id == "mom_v1"
        assert strategy.lookback_window == 5
        assert strategy.threshold == 0.02

    def test_momentum_strategy_on_data(self, sample_ohlcv_data: list[OHLCV]) -> None:
        """MomentumStrategy should accumulate data on on_data calls."""
        strategy = MomentumStrategy(
            strategy_id="mom_v1",
            config={"lookback_window": 5},
        )
        for bar in sample_ohlcv_data:
            strategy.on_data(bar)

        assert len(strategy.data_buffer) == len(sample_ohlcv_data)

    def test_momentum_strategy_generates_long_signal(
        self, sample_ohlcv_data: list[OHLCV]
    ) -> None:
        """MomentumStrategy should generate LONG signal on upward momentum."""
        strategy = MomentumStrategy(
            strategy_id="mom_v1",
            config={"lookback_window": 5, "threshold": 0.01},
        )
        # Feed data
        for bar in sample_ohlcv_data:
            strategy.on_data(bar)

        # Generate signal with features
        features = {"returns": 0.05}  # 5% return - above threshold
        signal = strategy.generate_signal(features)

        assert signal is not None
        assert signal.direction == SignalDirection.LONG
        assert signal.strategy_id == "mom_v1"

    def test_momentum_strategy_generates_short_signal(self) -> None:
        """MomentumStrategy should generate SHORT signal on downward momentum."""
        strategy = MomentumStrategy(
            strategy_id="mom_v1",
            config={"lookback_window": 5, "threshold": 0.01},
        )
        # Negative momentum
        features = {"returns": -0.05}  # -5% return
        signal = strategy.generate_signal(features)

        assert signal is not None
        assert signal.direction == SignalDirection.SHORT

    def test_momentum_strategy_generates_neutral_signal(self) -> None:
        """MomentumStrategy should generate NEUTRAL signal when within threshold."""
        strategy = MomentumStrategy(
            strategy_id="mom_v1",
            config={"lookback_window": 5, "threshold": 0.05},
        )
        # Small movement within threshold
        features = {"returns": 0.02}  # 2% - below 5% threshold
        signal = strategy.generate_signal(features)

        assert signal is not None
        assert signal.direction == SignalDirection.NEUTRAL

    def test_momentum_strategy_signal_has_confidence(self) -> None:
        """MomentumStrategy signal should have confidence proportional to momentum."""
        strategy = MomentumStrategy(
            strategy_id="mom_v1",
            config={"lookback_window": 5, "threshold": 0.01},
        )
        features = {"returns": 0.10}  # 10% return
        signal = strategy.generate_signal(features)

        assert signal is not None
        assert 0 <= signal.confidence <= 1
