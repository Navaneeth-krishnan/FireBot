"""Model versioning for ML strategy artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelVersion:
    """Immutable record of a model version.

    Attributes:
        model_name: Name of the model
        version: Semantic version string
        metadata: Arbitrary metadata (accuracy, framework, features, etc.)
        created_at: Timestamp when registered
    """

    model_name: str
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelVersionManager:
    """Manages model versions and artifacts on disk.

    Provides registration, listing, and artifact storage for ML model
    versions. Each model can have multiple versions with metadata and
    serialized artifacts.

    Storage layout:
        storage_path/
            {model_name}/
                versions.jsonl     # version metadata
                {version}/
                    artifact.bin   # serialized model

    Example:
        manager = ModelVersionManager(storage_path=Path("./models"))
        version = manager.register("my_model", "1.0.0", {"accuracy": 0.85})
        manager.save_artifact(version, model_bytes)
        loaded = manager.load_artifact(version)
    """

    def __init__(self, storage_path: Path) -> None:
        """Initialize version manager.

        Args:
            storage_path: Base directory for model storage
        """
        self.storage_path = storage_path
        self._versions: dict[str, list[ModelVersion]] = {}

        # Load existing versions from disk
        self._load_existing()

    def register(
        self,
        model_name: str,
        version: str,
        metadata: dict[str, Any],
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            model_name: Model identifier
            version: Semantic version string
            metadata: Version metadata (accuracy, features, etc.)

        Returns:
            The created ModelVersion
        """
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            metadata=dict(metadata),
        )

        if model_name not in self._versions:
            self._versions[model_name] = []

        self._versions[model_name].append(model_version)
        self._persist_version(model_version)

        return model_version

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        """List all versions for a model.

        Args:
            model_name: Model identifier

        Returns:
            List of ModelVersion objects
        """
        return list(self._versions.get(model_name, []))

    def get_latest(self, model_name: str) -> ModelVersion | None:
        """Get the most recently registered version.

        Args:
            model_name: Model identifier

        Returns:
            Latest ModelVersion or None if no versions exist
        """
        versions = self._versions.get(model_name, [])
        if not versions:
            return None
        return versions[-1]

    def get_version(self, model_name: str, version: str) -> ModelVersion | None:
        """Get a specific version by name.

        Args:
            model_name: Model identifier
            version: Version string to find

        Returns:
            ModelVersion if found, None otherwise
        """
        for v in self._versions.get(model_name, []):
            if v.version == version:
                return v
        return None

    def save_artifact(self, version: ModelVersion, data: bytes) -> Path:
        """Save model artifact to disk.

        Args:
            version: ModelVersion to associate with
            data: Serialized model bytes

        Returns:
            Path where artifact was saved
        """
        artifact_dir = self.storage_path / version.model_name / version.version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "artifact.bin"
        artifact_path.write_bytes(data)
        return artifact_path

    def load_artifact(self, version: ModelVersion) -> bytes | None:
        """Load model artifact from disk.

        Args:
            version: ModelVersion to load

        Returns:
            Artifact bytes or None if not found
        """
        artifact_path = (
            self.storage_path / version.model_name / version.version / "artifact.bin"
        )
        if not artifact_path.exists():
            return None
        return artifact_path.read_bytes()

    def _persist_version(self, version: ModelVersion) -> None:
        """Append version metadata to disk."""
        model_dir = self.storage_path / version.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        versions_file = model_dir / "versions.jsonl"

        record = {
            "model_name": version.model_name,
            "version": version.version,
            "metadata": version.metadata,
            "created_at": version.created_at.isoformat(),
        }
        with open(versions_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _load_existing(self) -> None:
        """Load existing version metadata from disk."""
        if not self.storage_path.exists():
            return

        for model_dir in self.storage_path.iterdir():
            if not model_dir.is_dir():
                continue
            versions_file = model_dir / "versions.jsonl"
            if not versions_file.exists():
                continue

            model_name = model_dir.name
            self._versions[model_name] = []

            with open(versions_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._versions[model_name].append(
                        ModelVersion(
                            model_name=record["model_name"],
                            version=record["version"],
                            metadata=record.get("metadata", {}),
                            created_at=datetime.fromisoformat(record["created_at"]),
                        )
                    )
