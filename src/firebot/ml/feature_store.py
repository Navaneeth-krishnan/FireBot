"""Feature store for ML strategy feature management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class FeatureSet:
    """Definition of a named feature set.

    Attributes:
        name: Unique identifier for this feature set
        features: List of feature names included
        version: Semantic version string
        metadata: Optional additional metadata
    """

    name: str
    features: list[str]
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FeatureRecord:
    """Internal record of stored feature values."""

    timestamp: datetime
    symbol: str
    values: dict[str, Any]


class FeatureStore:
    """In-memory feature store for ML strategies.

    Stores named feature sets and their computed values, supporting
    time-series retrieval per symbol. Designed to sit between the
    FeaturePipeline and MLStrategy.

    Example:
        store = FeatureStore()
        store.register(FeatureSet(name="tech_v1", features=["sma_20"], version="1.0"))
        store.put("tech_v1", "AAPL", timestamp, {"sma_20": 150.0})
        latest = store.get_latest("tech_v1", "AAPL")
    """

    def __init__(self) -> None:
        self._feature_sets: dict[str, FeatureSet] = {}
        self._data: dict[str, list[_FeatureRecord]] = {}  # keyed by "set_name:symbol"

    @property
    def count(self) -> int:
        """Number of registered feature sets."""
        return len(self._feature_sets)

    def register(self, feature_set: FeatureSet) -> None:
        """Register a feature set definition.

        Args:
            feature_set: Feature set to register
        """
        self._feature_sets[feature_set.name] = feature_set

    def get(self, name: str) -> FeatureSet | None:
        """Get a registered feature set by name.

        Args:
            name: Feature set name

        Returns:
            FeatureSet if found, None otherwise
        """
        return self._feature_sets.get(name)

    def put(
        self,
        feature_set_name: str,
        symbol: str,
        timestamp: datetime,
        values: dict[str, Any],
    ) -> None:
        """Store feature values for a symbol at a timestamp.

        Args:
            feature_set_name: Name of the registered feature set
            symbol: Trading symbol
            timestamp: Observation timestamp
            values: Feature name -> value mapping
        """
        key = f"{feature_set_name}:{symbol}"
        if key not in self._data:
            self._data[key] = []

        self._data[key].append(
            _FeatureRecord(timestamp=timestamp, symbol=symbol, values=dict(values))
        )
        # Keep sorted by timestamp
        self._data[key].sort(key=lambda r: r.timestamp)

    def get_latest(
        self,
        feature_set_name: str,
        symbol: str,
    ) -> dict[str, Any] | None:
        """Get the most recent feature values for a symbol.

        Args:
            feature_set_name: Feature set name
            symbol: Trading symbol

        Returns:
            Feature values dict or None if no data
        """
        key = f"{feature_set_name}:{symbol}"
        records = self._data.get(key)
        if not records:
            return None
        return dict(records[-1].values)

    def get_history(
        self,
        feature_set_name: str,
        symbol: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get historical feature values for a symbol.

        Args:
            feature_set_name: Feature set name
            symbol: Trading symbol
            limit: Maximum number of records (most recent first)

        Returns:
            List of feature value dicts, most recent first
        """
        key = f"{feature_set_name}:{symbol}"
        records = self._data.get(key, [])

        # Return most recent first
        reversed_records = list(reversed(records))

        if limit is not None:
            reversed_records = reversed_records[:limit]

        return [dict(r.values) for r in reversed_records]

    def list_feature_sets(self) -> list[str]:
        """List all registered feature set names.

        Returns:
            List of feature set names
        """
        return list(self._feature_sets.keys())
