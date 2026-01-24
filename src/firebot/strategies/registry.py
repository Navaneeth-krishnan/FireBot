"""Strategy plugin registry for discovering and loading strategies."""

from typing import Any, Callable, TypeVar

from firebot.strategies.base import Strategy

T = TypeVar("T", bound=Strategy)


class StrategyRegistry:
    """Registry for strategy plugins.

    Allows strategies to be registered by name and instantiated
    from configuration.

    Example:
        registry = StrategyRegistry()

        @registry.register("momentum")
        class MomentumStrategy(Strategy):
            ...

        # Later, create instance from config
        strategy = registry.create("momentum", "my_strat", {"window": 20})
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._strategies: dict[str, type[Strategy]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a strategy class.

        Args:
            name: Unique name for the strategy

        Returns:
            Decorator function

        Raises:
            ValueError: If name is already registered
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._strategies:
                raise ValueError(f"Strategy '{name}' is already registered")
            self._strategies[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[Strategy]:
        """Get a registered strategy class by name.

        Args:
            name: Strategy name

        Returns:
            Strategy class

        Raises:
            KeyError: If strategy is not registered
        """
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' is not registered")
        return self._strategies[name]

    def create(
        self,
        name: str,
        strategy_id: str,
        config: dict[str, Any],
    ) -> Strategy:
        """Create a strategy instance from registry.

        Args:
            name: Registered strategy name
            strategy_id: Unique ID for this instance
            config: Configuration parameters

        Returns:
            Configured strategy instance
        """
        strategy_cls = self.get(name)
        return strategy_cls(strategy_id=strategy_id, config=config)

    def list_strategies(self) -> list[str]:
        """List all registered strategy names.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    def unregister(self, name: str) -> None:
        """Remove a strategy from the registry.

        Args:
            name: Strategy name to remove

        Raises:
            KeyError: If strategy is not registered
        """
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' is not registered")
        del self._strategies[name]


# Global registry instance
default_registry = StrategyRegistry()
