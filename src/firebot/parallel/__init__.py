"""Parallel execution module using Ray."""

from firebot.parallel.actor import StrategyActor
from firebot.parallel.runner import ParallelRunner

__all__ = ["StrategyActor", "ParallelRunner"]
