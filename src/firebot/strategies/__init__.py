"""Strategy engine with plugin architecture."""

from firebot.strategies.base import Strategy
from firebot.strategies.registry import StrategyRegistry, default_registry
from firebot.strategies.momentum import MomentumStrategy

__all__ = [
    "Strategy",
    "StrategyRegistry",
    "default_registry",
    "MomentumStrategy",
]
