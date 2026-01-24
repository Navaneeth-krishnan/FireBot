"""Configuration loading and validation for FireBot."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataSourceConfig(BaseModel):
    """Configuration for a data source."""

    type: str
    path: str | None = None
    symbols: list[str] = Field(default_factory=list)


class StrategyConfig(BaseModel):
    """Configuration for a strategy instance."""

    name: str
    class_name: str = Field(alias="class")
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    max_daily_loss_pct: float = 3.0
    auto_disable_on_breach: bool = True


class PortfolioConfig(BaseModel):
    """Portfolio configuration."""

    initial_capital: float = 100000.0
    currency: str = "USD"


class ExecutionConfig(BaseModel):
    """Execution engine configuration."""

    fill_model: str = "instant"
    slippage_bps: float = 0.0
    commission_per_trade: float = 0.0


class AppConfig(BaseModel):
    """Main application configuration."""

    name: str = "FireBot"
    log_level: str = "INFO"
    data_dir: str = "./data"
    model_dir: str = "./models"


class FireBotConfig(BaseModel):
    """Complete FireBot configuration."""

    app: AppConfig = Field(default_factory=AppConfig)
    data_sources: list[DataSourceConfig] = Field(default_factory=list)
    strategies: list[StrategyConfig] = Field(default_factory=list)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    class Config:
        populate_by_name = True


def load_config(path: Path | str) -> FireBotConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Validated FireBotConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        raw_config = yaml.safe_load(f)

    return FireBotConfig(**raw_config)


def save_config(config: FireBotConfig, path: Path | str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        path: Path to write YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.model_dump(by_alias=True), f, default_flow_style=False)
