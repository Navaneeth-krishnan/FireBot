"""Tests for Signal Aggregation Layer."""

from datetime import datetime, timezone
from typing import Any

import pytest

from firebot.core.models import Signal, SignalDirection
from firebot.aggregation.aggregator import (
    SignalAggregator,
    MajorityVoteAggregator,
    WeightedAverageAggregator,
    UnanimityAggregator,
)


def _make_signal(
    direction: SignalDirection,
    confidence: float = 0.5,
    strategy_id: str = "test",
    symbol: str = "AAPL",
) -> Signal:
    """Helper to create test signals."""
    return Signal(
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        strategy_id=strategy_id,
    )


class TestMajorityVoteAggregator:
    """Tests for majority vote signal aggregation."""

    def test_all_long(self) -> None:
        """All LONG signals should aggregate to LONG."""
        agg = MajorityVoteAggregator(aggregator_id="mv_1")
        signals = [
            _make_signal(SignalDirection.LONG, strategy_id="s1"),
            _make_signal(SignalDirection.LONG, strategy_id="s2"),
            _make_signal(SignalDirection.LONG, strategy_id="s3"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.direction == SignalDirection.LONG

    def test_majority_long(self) -> None:
        """Majority LONG should win."""
        agg = MajorityVoteAggregator(aggregator_id="mv_1")
        signals = [
            _make_signal(SignalDirection.LONG, strategy_id="s1"),
            _make_signal(SignalDirection.LONG, strategy_id="s2"),
            _make_signal(SignalDirection.SHORT, strategy_id="s3"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.direction == SignalDirection.LONG

    def test_majority_short(self) -> None:
        """Majority SHORT should win."""
        agg = MajorityVoteAggregator(aggregator_id="mv_1")
        signals = [
            _make_signal(SignalDirection.SHORT, strategy_id="s1"),
            _make_signal(SignalDirection.SHORT, strategy_id="s2"),
            _make_signal(SignalDirection.LONG, strategy_id="s3"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.direction == SignalDirection.SHORT

    def test_tie_returns_neutral(self) -> None:
        """Tie should resolve to NEUTRAL (no action)."""
        agg = MajorityVoteAggregator(aggregator_id="mv_1")
        signals = [
            _make_signal(SignalDirection.LONG, strategy_id="s1"),
            _make_signal(SignalDirection.SHORT, strategy_id="s2"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.direction == SignalDirection.NEUTRAL

    def test_empty_signals_returns_none(self) -> None:
        """No signals should return None."""
        agg = MajorityVoteAggregator(aggregator_id="mv_1")
        result = agg.aggregate([])
        assert result is None

    def test_confidence_is_vote_proportion(self) -> None:
        """Confidence should reflect the proportion of votes for the winner."""
        agg = MajorityVoteAggregator(aggregator_id="mv_1")
        signals = [
            _make_signal(SignalDirection.LONG, strategy_id="s1"),
            _make_signal(SignalDirection.LONG, strategy_id="s2"),
            _make_signal(SignalDirection.SHORT, strategy_id="s3"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.confidence == pytest.approx(2 / 3, abs=0.01)


class TestWeightedAverageAggregator:
    """Tests for weighted average aggregation."""

    def test_equal_weights(self) -> None:
        """Equal weights should behave like simple average."""
        weights = {"s1": 1.0, "s2": 1.0, "s3": 1.0}
        agg = WeightedAverageAggregator(aggregator_id="wa_1", weights=weights)
        signals = [
            _make_signal(SignalDirection.LONG, confidence=0.8, strategy_id="s1"),
            _make_signal(SignalDirection.LONG, confidence=0.6, strategy_id="s2"),
            _make_signal(SignalDirection.SHORT, confidence=0.3, strategy_id="s3"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        # Weighted: (0.8*1 + 0.6*1 + (-0.3)*1) / 3 = 1.1/3 = 0.367 -> LONG
        assert result.direction == SignalDirection.LONG

    def test_high_weight_dominates(self) -> None:
        """Strategy with high weight should dominate."""
        weights = {"s1": 10.0, "s2": 1.0}
        agg = WeightedAverageAggregator(aggregator_id="wa_1", weights=weights)
        signals = [
            _make_signal(SignalDirection.SHORT, confidence=0.8, strategy_id="s1"),
            _make_signal(SignalDirection.LONG, confidence=0.9, strategy_id="s2"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.direction == SignalDirection.SHORT

    def test_threshold_filtering(self) -> None:
        """Weak signal below threshold should return NEUTRAL."""
        weights = {"s1": 1.0, "s2": 1.0}
        agg = WeightedAverageAggregator(
            aggregator_id="wa_1", weights=weights, threshold=0.5
        )
        signals = [
            _make_signal(SignalDirection.LONG, confidence=0.3, strategy_id="s1"),
            _make_signal(SignalDirection.SHORT, confidence=0.2, strategy_id="s2"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        # Weighted score is weak -> below threshold -> NEUTRAL
        assert result.direction == SignalDirection.NEUTRAL

    def test_unknown_strategy_gets_default_weight(self) -> None:
        """Strategy not in weights dict should use default weight."""
        weights = {"s1": 2.0}
        agg = WeightedAverageAggregator(
            aggregator_id="wa_1", weights=weights, default_weight=1.0
        )
        signals = [
            _make_signal(SignalDirection.LONG, confidence=0.5, strategy_id="s1"),
            _make_signal(SignalDirection.LONG, confidence=0.5, strategy_id="s_unknown"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.direction == SignalDirection.LONG


class TestUnanimityAggregator:
    """Tests for unanimity-required aggregation."""

    def test_unanimous_long(self) -> None:
        """All LONG should produce LONG."""
        agg = UnanimityAggregator(aggregator_id="un_1")
        signals = [
            _make_signal(SignalDirection.LONG, strategy_id="s1"),
            _make_signal(SignalDirection.LONG, strategy_id="s2"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.direction == SignalDirection.LONG

    def test_not_unanimous_returns_neutral(self) -> None:
        """Mixed signals should return NEUTRAL."""
        agg = UnanimityAggregator(aggregator_id="un_1")
        signals = [
            _make_signal(SignalDirection.LONG, strategy_id="s1"),
            _make_signal(SignalDirection.SHORT, strategy_id="s2"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.direction == SignalDirection.NEUTRAL

    def test_single_signal_passes(self) -> None:
        """Single signal should pass through."""
        agg = UnanimityAggregator(aggregator_id="un_1")
        signals = [_make_signal(SignalDirection.SHORT, strategy_id="s1")]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.direction == SignalDirection.SHORT

    def test_neutrals_excluded(self) -> None:
        """NEUTRAL signals should be excluded from unanimity check."""
        agg = UnanimityAggregator(aggregator_id="un_1")
        signals = [
            _make_signal(SignalDirection.LONG, strategy_id="s1"),
            _make_signal(SignalDirection.NEUTRAL, strategy_id="s2"),
            _make_signal(SignalDirection.LONG, strategy_id="s3"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert result.direction == SignalDirection.LONG


class TestSignalAggregatorInterface:
    """Tests for the aggregator base interface."""

    def test_aggregated_signal_has_metadata(self) -> None:
        """Aggregated signal should include source strategy info."""
        agg = MajorityVoteAggregator(aggregator_id="mv_1")
        signals = [
            _make_signal(SignalDirection.LONG, strategy_id="s1"),
            _make_signal(SignalDirection.LONG, strategy_id="s2"),
        ]
        result = agg.aggregate(signals)
        assert result is not None
        assert "source_strategies" in result.metadata
        assert result.strategy_id == "mv_1"
