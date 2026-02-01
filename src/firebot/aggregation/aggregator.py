"""Signal aggregation strategies for ensemble trading.

Combines signals from multiple strategies into a single actionable
signal using various aggregation methods: majority vote, weighted
average, and unanimity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from firebot.core.models import Signal, SignalDirection


class SignalAggregator(ABC):
    """Base class for signal aggregation.

    Aggregators take a list of signals from multiple strategies and
    produce a single combined signal.
    """

    def __init__(self, aggregator_id: str) -> None:
        self.aggregator_id = aggregator_id

    @abstractmethod
    def aggregate(self, signals: list[Signal]) -> Signal | None:
        """Aggregate multiple signals into one.

        Args:
            signals: List of signals from different strategies

        Returns:
            Aggregated signal, or None if no signals provided
        """
        ...

    def _make_result_signal(
        self,
        direction: SignalDirection,
        confidence: float,
        signals: list[Signal],
    ) -> Signal:
        """Create an aggregated result signal with metadata."""
        symbol = signals[0].symbol if signals else "UNKNOWN"
        return Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            direction=direction,
            confidence=min(max(confidence, 0.0), 1.0),
            strategy_id=self.aggregator_id,
            metadata={
                "source_strategies": [s.strategy_id for s in signals],
                "aggregation_method": self.__class__.__name__,
            },
        )


class MajorityVoteAggregator(SignalAggregator):
    """Aggregation by majority vote.

    Each signal gets one vote. The direction with the most votes wins.
    Ties resolve to NEUTRAL. Confidence is the proportion of votes
    for the winning direction.

    Example:
        agg = MajorityVoteAggregator(aggregator_id="majority_1")
        result = agg.aggregate([signal_a, signal_b, signal_c])
    """

    def aggregate(self, signals: list[Signal]) -> Signal | None:
        if not signals:
            return None

        votes = Counter(s.direction for s in signals)
        total = len(signals)

        long_count = votes.get(SignalDirection.LONG, 0)
        short_count = votes.get(SignalDirection.SHORT, 0)

        if long_count > short_count:
            direction = SignalDirection.LONG
            confidence = long_count / total
        elif short_count > long_count:
            direction = SignalDirection.SHORT
            confidence = short_count / total
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.0

        return self._make_result_signal(direction, confidence, signals)


class WeightedAverageAggregator(SignalAggregator):
    """Aggregation by weighted average of signal scores.

    Each signal contributes: direction_value * confidence * weight.
    LONG = +1, SHORT = -1, NEUTRAL = 0. The weighted average
    determines direction; if below threshold, returns NEUTRAL.

    Args:
        weights: Dict of strategy_id -> weight
        default_weight: Weight for unknown strategies
        threshold: Minimum absolute score to generate a directional signal

    Example:
        agg = WeightedAverageAggregator(
            aggregator_id="weighted_1",
            weights={"momentum": 2.0, "mean_revert": 1.0},
        )
        result = agg.aggregate(signals)
    """

    def __init__(
        self,
        aggregator_id: str,
        weights: dict[str, float],
        default_weight: float = 1.0,
        threshold: float = 0.0,
    ) -> None:
        super().__init__(aggregator_id)
        self.weights = dict(weights)
        self.default_weight = default_weight
        self.threshold = threshold

    def aggregate(self, signals: list[Signal]) -> Signal | None:
        if not signals:
            return None

        weighted_sum = 0.0
        total_weight = 0.0

        for signal in signals:
            weight = self.weights.get(signal.strategy_id, self.default_weight)
            direction_value = signal.direction.value  # +1, -1, 0
            weighted_sum += direction_value * signal.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return self._make_result_signal(SignalDirection.NEUTRAL, 0.0, signals)

        avg_score = weighted_sum / total_weight

        if abs(avg_score) < self.threshold:
            direction = SignalDirection.NEUTRAL
        elif avg_score > 0:
            direction = SignalDirection.LONG
        else:
            direction = SignalDirection.SHORT

        confidence = min(abs(avg_score), 1.0)
        return self._make_result_signal(direction, confidence, signals)


class UnanimityAggregator(SignalAggregator):
    """Aggregation requiring unanimous agreement.

    Only produces a directional signal when all non-NEUTRAL strategies
    agree on the same direction. Otherwise returns NEUTRAL.

    Example:
        agg = UnanimityAggregator(aggregator_id="unanimity_1")
        result = agg.aggregate(signals)  # Only LONG if ALL say LONG
    """

    def aggregate(self, signals: list[Signal]) -> Signal | None:
        if not signals:
            return None

        # Filter out NEUTRAL signals
        directional = [s for s in signals if s.direction != SignalDirection.NEUTRAL]

        if not directional:
            return self._make_result_signal(SignalDirection.NEUTRAL, 0.0, signals)

        directions = {s.direction for s in directional}

        if len(directions) == 1:
            direction = directions.pop()
            avg_confidence = sum(s.confidence for s in directional) / len(directional)
            return self._make_result_signal(direction, avg_confidence, signals)

        return self._make_result_signal(SignalDirection.NEUTRAL, 0.0, signals)
