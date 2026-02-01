"""Signal aggregation layer for ensemble strategies."""

from firebot.aggregation.aggregator import (
    SignalAggregator,
    MajorityVoteAggregator,
    WeightedAverageAggregator,
    UnanimityAggregator,
)

__all__ = [
    "SignalAggregator",
    "MajorityVoteAggregator",
    "WeightedAverageAggregator",
    "UnanimityAggregator",
]
