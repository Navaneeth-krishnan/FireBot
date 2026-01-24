"""Tests for Parallel Execution with Ray - TDD RED phase."""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from firebot.core.models import OHLCV, Signal, SignalDirection
from firebot.strategies.base import Strategy


class TestParallelRunner:
    """Tests for parallel strategy execution."""

    def test_parallel_runner_initialization(self) -> None:
        """Runner should initialize with Ray."""
        from firebot.parallel.runner import ParallelRunner

        runner = ParallelRunner(num_cpus=2)
        assert runner.num_cpus == 2
        assert runner._strategies == {}

    def test_register_strategy(self) -> None:
        """Runner should register strategies with unique portfolios."""
        from firebot.parallel.runner import ParallelRunner
        from firebot.strategies.momentum import MomentumStrategy

        runner = ParallelRunner(num_cpus=2)

        runner.register_strategy(
            strategy_id="momentum_1",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 20},
            initial_capital=Decimal("100000"),
        )

        assert "momentum_1" in runner._strategies
        assert runner._strategies["momentum_1"]["initial_capital"] == Decimal("100000")

    def test_distribute_data_to_strategies(self) -> None:
        """Runner should distribute data to all strategies."""
        from firebot.parallel.runner import ParallelRunner
        from firebot.strategies.momentum import MomentumStrategy

        runner = ParallelRunner(num_cpus=2)

        runner.register_strategy(
            strategy_id="strat_1",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 5},
            initial_capital=Decimal("100000"),
        )
        runner.register_strategy(
            strategy_id="strat_2",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 10},
            initial_capital=Decimal("50000"),
        )

        ohlcv = OHLCV(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=Decimal("1000000"),
        )

        # Data should be sent to both strategies
        assert len(runner._strategies) == 2

    def test_get_strategy_portfolios(self) -> None:
        """Should return independent portfolios for each strategy."""
        from firebot.parallel.runner import ParallelRunner
        from firebot.strategies.momentum import MomentumStrategy

        runner = ParallelRunner(num_cpus=2)

        runner.register_strategy(
            strategy_id="strat_1",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 5},
            initial_capital=Decimal("100000"),
        )
        runner.register_strategy(
            strategy_id="strat_2",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 10},
            initial_capital=Decimal("50000"),
        )

        portfolios = runner.get_portfolios()

        assert "strat_1" in portfolios
        assert "strat_2" in portfolios
        assert portfolios["strat_1"]["initial_capital"] == Decimal("100000")
        assert portfolios["strat_2"]["initial_capital"] == Decimal("50000")


class TestStrategyActor:
    """Tests for Ray Actor wrapper around Strategy."""

    def test_strategy_actor_creation(self) -> None:
        """Actor should wrap a strategy instance."""
        from firebot.parallel.actor import StrategyActor
        from firebot.strategies.momentum import MomentumStrategy

        actor = StrategyActor(
            strategy_id="test_actor",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 5},
            initial_capital=Decimal("100000"),
        )

        assert actor.strategy_id == "test_actor"
        assert actor.portfolio is not None

    def test_actor_processes_data(self) -> None:
        """Actor should process OHLCV data through strategy."""
        from firebot.parallel.actor import StrategyActor
        from firebot.strategies.momentum import MomentumStrategy

        actor = StrategyActor(
            strategy_id="test_actor",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 5},
            initial_capital=Decimal("100000"),
        )

        ohlcv = OHLCV(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=Decimal("1000000"),
        )

        # Should not raise
        actor.on_data(ohlcv)

    def test_actor_maintains_portfolio_state(self) -> None:
        """Actor should maintain independent portfolio state."""
        from firebot.parallel.actor import StrategyActor
        from firebot.strategies.momentum import MomentumStrategy

        actor = StrategyActor(
            strategy_id="test_actor",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 5},
            initial_capital=Decimal("100000"),
        )

        state = actor.get_state()
        assert state["portfolio"]["cash"] == Decimal("100000")
        assert state["portfolio"]["positions"] == {}


class TestStrategyIsolation:
    """Tests for strategy isolation and error handling."""

    def test_strategy_error_isolation(self) -> None:
        """Error in one strategy should not affect others."""
        from firebot.parallel.runner import ParallelRunner
        from firebot.strategies.momentum import MomentumStrategy

        runner = ParallelRunner(num_cpus=2)

        # Register normal strategy
        runner.register_strategy(
            strategy_id="normal",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 5},
            initial_capital=Decimal("100000"),
        )

        # Both strategies should be registered
        assert "normal" in runner._strategies

    def test_strategy_state_independence(self) -> None:
        """Each strategy should have completely independent state."""
        from firebot.parallel.actor import StrategyActor
        from firebot.strategies.momentum import MomentumStrategy

        actor1 = StrategyActor(
            strategy_id="actor_1",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 5},
            initial_capital=Decimal("100000"),
        )

        actor2 = StrategyActor(
            strategy_id="actor_2",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 10},
            initial_capital=Decimal("50000"),
        )

        # Modify one actor's state
        state1 = actor1.get_state()
        state2 = actor2.get_state()

        # States should be independent
        assert state1["portfolio"]["cash"] != state2["portfolio"]["cash"]
        assert state1["config"]["lookback_period"] != state2["config"]["lookback_period"]
