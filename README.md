# FireBot

A modular, Python-first paper trading platform for ML-based strategy experimentation.

FireBot is a research-focused platform designed for experimentation, learning, and strategy validation — not production trading. Its design philosophy: *"Treat strategies as models, markets as datasets, and trading as inference."*

## Features

- **Plugin strategy system** — define new strategies by subclassing `Strategy` and registering them with a decorator
- **ML strategies** — first-class PyTorch support with a Transformer-based strategy, feature store, and model versioning
- **Paper trading engine** — order simulation with configurable slippage, stop-loss/take-profit, and risk controls that auto-disable on max drawdown breach
- **Signal aggregation** — combine signals from multiple strategies via majority vote, weighted average, or unanimity
- **Deterministic backtesting** — reproducible backtests with forward testing support
- **Parallel execution** — run multiple strategies simultaneously with independent virtual portfolios via Ray
- **Metrics and monitoring** — Sharpe, Sortino, max drawdown, win rate, profit factor — exported to Prometheus/InfluxDB with Grafana dashboards
- **Feature engineering** — technical indicators (SMA, EMA, RSI, Bollinger Bands) computed with Polars

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Core dependencies only
uv sync

# With development tools
uv sync --extra dev

# Everything (ML, parallel, metrics, visualization, dev)
uv sync --extra all
```

Available extras:

| Extra      | Packages                              |
|------------|---------------------------------------|
| `ml`       | PyTorch                               |
| `parallel` | Ray                                   |
| `metrics`  | Prometheus client, InfluxDB client    |
| `viz`      | Matplotlib                            |
| `dev`      | pytest, black, ruff, mypy             |
| `all`      | All of the above                      |

## Quick Start

### Defining a Strategy

Create a new strategy by subclassing `Strategy` and registering it:

```python
from firebot.strategies.base import Strategy
from firebot.strategies.registry import default_registry
from firebot.core.models import OHLCV, Signal, SignalDirection

@default_registry.register("my_strategy")
class MyStrategy(Strategy):
    def on_data(self, data: OHLCV) -> None:
        self.data_buffer.append(data)

    def generate_signal(self, features: dict) -> Signal | None:
        if features.get("sma_20", 0) > features.get("sma_50", 0):
            return Signal(
                timestamp=self.data_buffer[-1].timestamp,
                symbol=self.data_buffer[-1].symbol,
                direction=SignalDirection.LONG,
                confidence=0.8,
                strategy_id=self.strategy_id,
            )
        return None
```

Then instantiate it from the registry:

```python
strategy = default_registry.create("my_strategy", "my_strat_01", {"window": 20})
```

### Configuration

FireBot uses YAML configuration validated through Pydantic:

```yaml
app:
  name: FireBot
  log_level: INFO
  data_dir: ./data

data_sources:
  - type: csv
    path: ./data/prices.csv
    symbols: ["AAPL", "GOOGL"]

strategies:
  - name: momentum_1
    class: momentum
    params:
      lookback_window: 20
      threshold: 0.02

portfolio:
  initial_capital: 100000.0
  currency: USD

risk:
  max_position_size_pct: 5.0
  max_drawdown_pct: 10.0
  auto_disable_on_breach: true

execution:
  fill_model: instant
  slippage_bps: 5.0
  commission_per_trade: 1.0
```

Load configuration in code:

```python
from firebot.core.config import load_config

config = load_config("configs/my_config.yaml")
```

## Architecture

```
DataSource → FeaturePipeline → Strategy → Signal → Aggregator
    → PaperTradingEngine → PortfolioSimulator → MetricsEngine → Exporters
```

| Module | Purpose |
|---|---|
| `core/` | Pydantic data models (OHLCV, Signal, Order, Position, Portfolio) and config loading |
| `data/sources/` | Pluggable data source interface (CSV implementation included) |
| `features/` | Feature pipeline with technical indicators via Polars |
| `strategies/` | Strategy base class, plugin registry, and reference momentum strategy |
| `ml/` | PyTorch ML strategies — Transformer model, feature store, model versioning |
| `aggregation/` | Ensemble signal combination (majority vote, weighted average, unanimity) |
| `execution/` | Paper trading engine with order simulation and portfolio tracking |
| `backtesting/` | Deterministic backtesting and forward testing engines |
| `metrics/` | Performance calculators and Prometheus/InfluxDB exporters |
| `parallel/` | Multi-strategy parallel execution via Ray actors |
| `visualization/` | Matplotlib charts and Grafana dashboard templates |

All data models use `Decimal` for financial values and `frozen=True` for immutability.

## Docker

```bash
# Run FireBot
docker-compose up firebot

# Run tests
docker-compose --profile test run firebot-test

# Run with Prometheus + Grafana monitoring
docker-compose --profile monitoring up
```

The monitoring stack exposes:
- **Prometheus** at `http://localhost:9090`
- **Grafana** at `http://localhost:3000` (admin/firebot)

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/unit/test_execution.py

# Run with coverage
pytest tests/ --cov=src/firebot

# Lint
ruff check src/ tests/

# Format
black src/ tests/

# Type check
mypy src/
```

## License

MIT
