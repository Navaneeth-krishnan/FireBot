"""Parallel Runner for executing multiple strategies with Ray."""

from decimal import Decimal
from typing import Any

from firebot.core.models import OHLCV
from firebot.parallel.actor import StrategyActor
from firebot.strategies.base import Strategy


class ParallelRunner:
    """Orchestrates parallel execution of multiple strategies.

    Uses Ray for distributed execution with independent portfolios
    per strategy. Each strategy runs in isolation with its own
    portfolio state.

    Example:
        runner = ParallelRunner(num_cpus=4)
        runner.register_strategy(
            strategy_id="momentum_fast",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 10},
            initial_capital=Decimal("100000"),
        )
        runner.register_strategy(
            strategy_id="momentum_slow",
            strategy_class=MomentumStrategy,
            config={"lookback_period": 50},
            initial_capital=Decimal("100000"),
        )
        runner.run(data_source)
    """

    def __init__(
        self,
        num_cpus: int = 2,
        use_ray: bool = False,
    ) -> None:
        """Initialize parallel runner.

        Args:
            num_cpus: Number of CPUs to use for Ray
            use_ray: Whether to use Ray for parallelism (default False for testing)
        """
        self.num_cpus = num_cpus
        self.use_ray = use_ray
        self._strategies: dict[str, dict[str, Any]] = {}
        self._actors: dict[str, StrategyActor] = {}
        self._ray_initialized = False

    def _init_ray(self) -> None:
        """Initialize Ray if needed."""
        if self.use_ray and not self._ray_initialized:
            import ray

            if not ray.is_initialized():
                ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True)
            self._ray_initialized = True

    def register_strategy(
        self,
        strategy_id: str,
        strategy_class: type[Strategy],
        config: dict[str, Any],
        initial_capital: Decimal,
        max_drawdown_pct: Decimal = Decimal("0.10"),
        max_position_size_pct: Decimal = Decimal("0.05"),
    ) -> None:
        """Register a strategy for parallel execution.

        Args:
            strategy_id: Unique identifier for this strategy
            strategy_class: Strategy class to instantiate
            config: Configuration for the strategy
            initial_capital: Starting capital for portfolio
            max_drawdown_pct: Maximum drawdown threshold
            max_position_size_pct: Maximum position size
        """
        if strategy_id in self._strategies:
            raise ValueError(f"Strategy {strategy_id} already registered")

        self._strategies[strategy_id] = {
            "strategy_class": strategy_class,
            "config": config,
            "initial_capital": initial_capital,
            "max_drawdown_pct": max_drawdown_pct,
            "max_position_size_pct": max_position_size_pct,
        }

        # Create actor (non-Ray for now, Ray actor creation in run())
        self._actors[strategy_id] = StrategyActor(
            strategy_id=strategy_id,
            strategy_class=strategy_class,
            config=config,
            initial_capital=initial_capital,
            max_drawdown_pct=max_drawdown_pct,
            max_position_size_pct=max_position_size_pct,
        )

    def unregister_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from the runner.

        Args:
            strategy_id: Strategy to remove
        """
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
        if strategy_id in self._actors:
            del self._actors[strategy_id]

    def distribute_data(self, ohlcv: OHLCV) -> dict[str, Any]:
        """Distribute data to all registered strategies.

        Args:
            ohlcv: Market data to distribute

        Returns:
            Dict mapping strategy_id to any signals generated
        """
        results = {}
        for strategy_id, actor in self._actors.items():
            signal = actor.on_data(ohlcv)
            if signal:
                results[strategy_id] = signal
        return results

    def get_portfolios(self) -> dict[str, dict[str, Any]]:
        """Get current portfolio state for all strategies.

        Returns:
            Dict mapping strategy_id to portfolio info
        """
        portfolios = {}
        for strategy_id, config in self._strategies.items():
            if strategy_id in self._actors:
                state = self._actors[strategy_id].get_state()
                portfolios[strategy_id] = {
                    "initial_capital": config["initial_capital"],
                    **state["portfolio"],
                }
            else:
                portfolios[strategy_id] = {
                    "initial_capital": config["initial_capital"],
                }
        return portfolios

    def get_strategy_states(self) -> dict[str, dict[str, Any]]:
        """Get full state for all strategies.

        Returns:
            Dict mapping strategy_id to full state
        """
        return {
            strategy_id: actor.get_state()
            for strategy_id, actor in self._actors.items()
        }

    def run_backtest(
        self,
        data: list[OHLCV],
    ) -> dict[str, dict[str, Any]]:
        """Run backtest on historical data.

        Args:
            data: List of OHLCV data points in chronological order

        Returns:
            Final portfolio states for all strategies
        """
        for ohlcv in data:
            self.distribute_data(ohlcv)

        return self.get_strategy_states()

    def shutdown(self) -> None:
        """Shutdown Ray if initialized."""
        if self._ray_initialized:
            import ray

            ray.shutdown()
            self._ray_initialized = False
