# Technical Specifications
## FireBot - Modular ML-Based Paper Trading Platform

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Configuration (YAML)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Market Data Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  CSV Source │  │Parquet Source│ │  API Source │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Service                       │
│  - Format normalization                                         │
│  - Time-indexed storage                                         │
│  - Batch / streaming ingestion                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Feature Engineering Layer                      │
│  - Technical indicators                                         │
│  - Rolling statistics                                           │
│  - Custom ML features                                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                Strategy Engine (Plugin System)                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │
│  │Strategy 1 │  │Strategy 2 │  │Strategy 3 │  │Strategy N │     │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘     │
│                    (Parallel via Ray)                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Signal Aggregation Layer (Optional)                │
│  - Ensemble strategies                                          │
│  - Voting systems                                               │
│  - Confidence-weighted aggregation                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Paper Trading Engine                         │
│  - Order simulation                                             │
│  - Instant fills                                                │
│  - Stop-loss / Take-profit                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Portfolio Simulator                          │
│  - Position tracking                                            │
│  - PnL calculation                                              │
│  - Risk metrics                                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Metrics Engine                             │
│  - Performance metrics                                          │
│  - Risk metrics                                                 │
│  - Time-series export                                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Storage & Visualization                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │InfluxDB/    │  │   Model     │  │  Grafana    │              │
│  │Prometheus   │  │   Storage   │  │ Dashboards  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Models

### 2.1 OHLCV (Market Data)
```python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

@dataclass
class OHLCV:
    timestamp: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    resolution: str  # "1m", "5m", "15m", "1h", etc.
```

### 2.2 Signal (Strategy Output)
```python
from enum import Enum

class SignalDirection(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    strategy_id: str
    metadata: dict  # Optional additional info
```

### 2.3 Order (Execution Intent)
```python
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None  # None for market orders
    strategy_id: str
    status: str  # "pending", "filled", "cancelled"
```

### 2.4 Position (Current Holding)
```python
@dataclass
class Position:
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    strategy_id: str
```

### 2.5 Portfolio (Aggregate State)
```python
@dataclass
class Portfolio:
    strategy_id: str
    cash: Decimal
    positions: dict[str, Position]
    total_value: Decimal
    total_pnl: Decimal
    max_drawdown: Decimal
    high_water_mark: Decimal
```

---

## 3. Core Interfaces

### 3.1 Data Source Interface
```python
from abc import ABC, abstractmethod
from typing import Iterator

class DataSource(ABC):
    @abstractmethod
    def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        resolution: str
    ) -> Iterator[OHLCV]:
        """Fetch historical OHLCV data."""
        pass

    @abstractmethod
    def subscribe(self, symbol: str) -> None:
        """Subscribe to live data (future implementation)."""
        pass

    @abstractmethod
    def get_symbols(self) -> list[str]:
        """Get available symbols."""
        pass
```

### 3.2 Feature Pipeline Interface
```python
class FeaturePipeline(ABC):
    @abstractmethod
    def transform(self, raw_data: list[OHLCV]) -> dict:
        """Transform raw OHLCV data to features."""
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return list of feature names produced."""
        pass
```

### 3.3 Strategy Interface
```python
class Strategy(ABC):
    def __init__(self, strategy_id: str, config: dict):
        self.strategy_id = strategy_id
        self.config = config

    @abstractmethod
    def on_data(self, data: OHLCV) -> None:
        """Called when new data arrives."""
        pass

    @abstractmethod
    def generate_signal(self, features: dict) -> Signal | None:
        """Generate trading signal from features."""
        pass

    def on_fill(self, order: Order) -> None:
        """Called when order is filled (optional override)."""
        pass
```

### 3.4 Execution Engine Interface
```python
class ExecutionEngine(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit order for execution. Returns order ID."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Get current order status."""
        pass
```

### 3.5 Metrics Engine Interface
```python
class MetricsEngine(ABC):
    @abstractmethod
    def calculate_sharpe(self, returns: list[float], risk_free_rate: float = 0.0) -> float:
        pass

    @abstractmethod
    def calculate_sortino(self, returns: list[float], risk_free_rate: float = 0.0) -> float:
        pass

    @abstractmethod
    def calculate_max_drawdown(self, equity_curve: list[float]) -> float:
        pass

    @abstractmethod
    def calculate_win_rate(self, trades: list[dict]) -> float:
        pass

    @abstractmethod
    def export_to_timeseries_db(self, metrics: dict) -> None:
        pass
```

---

## 4. Configuration Schema

### 4.1 Main Configuration (config.yaml)
```yaml
# FireBot Configuration

app:
  name: "FireBot"
  log_level: "INFO"
  data_dir: "./data"
  model_dir: "./models"

data_sources:
  - type: "csv"
    path: "./data/historical"
    symbols: ["AAPL", "GOOGL", "MSFT"]
  - type: "parquet"
    path: "./data/parquet"

strategies:
  - name: "momentum_v1"
    class: "firebot.strategies.MomentumStrategy"
    enabled: true
    params:
      lookback_window: 20
      threshold: 0.02
  - name: "ml_transformer_v1"
    class: "firebot.strategies.TransformerStrategy"
    enabled: true
    params:
      model_path: "./models/transformer_v1.pt"
      sequence_length: 60

assets:
  - symbol: "AAPL"
    data_source: "csv"
  - symbol: "GOOGL"
    data_source: "csv"

risk:
  max_position_size_pct: 5.0
  max_drawdown_pct: 10.0
  max_daily_loss_pct: 3.0
  auto_disable_on_breach: true

portfolio:
  initial_capital: 100000.0
  currency: "USD"

execution:
  fill_model: "instant"  # or "realistic"
  slippage_bps: 0  # basis points
  commission_per_trade: 0.0

metrics:
  export_to: "influxdb"  # or "prometheus"
  influxdb:
    url: "http://localhost:8086"
    token: "${INFLUXDB_TOKEN}"
    org: "firebot"
    bucket: "trading_metrics"

parallel:
  enabled: true
  backend: "ray"
  num_workers: 4
```

---

## 5. Performance Requirements

### 5.1 Throughput
- Process minimum 1000 OHLCV bars per second per strategy
- Support 10+ strategies running in parallel
- Handle 100+ symbols simultaneously

### 5.2 Latency
- Signal generation: < 100ms per bar
- Order simulation: < 10ms per order
- Metrics calculation: < 1s for full portfolio

### 5.3 Storage
- Retain 5+ years of historical data per symbol
- Store all trades and portfolio snapshots
- Support incremental backups

---

## 6. Security Considerations

### 6.1 Secrets Management
- API keys stored in environment variables
- Never commit secrets to version control
- Use `.env` files for local development

### 6.2 Data Validation
- Validate all OHLCV data before processing
- Reject malformed or out-of-range values
- Log validation failures

### 6.3 Access Control
- Grafana authentication required
- InfluxDB token-based access
- No direct database exposure

---

## 7. Testing Requirements

### 7.1 Unit Tests
- All data models
- Feature calculations
- Strategy signal generation
- Metrics calculations

### 7.2 Integration Tests
- Data source connections
- End-to-end strategy execution
- Metrics export

### 7.3 Backtesting Validation
- Deterministic results with same seed
- Reproducible across runs
- No look-ahead bias

---

## 8. Deployment

### 8.1 Local Development
```bash
# Clone repository
git clone <repo>
cd firebot

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Start services (Docker)
docker-compose up -d
```

### 8.2 Docker Services
```yaml
# docker-compose.yml
version: "3.8"
services:
  influxdb:
    image: influxdb:2.7
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb2

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - influxdb

volumes:
  influxdb_data:
  grafana_data:
```

---

## 9. Directory Structure

```
firebot/
├── src/
│   └── firebot/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py          # Data models
│       │   ├── config.py          # Configuration loading
│       │   └── exceptions.py      # Custom exceptions
│       ├── data/
│       │   ├── __init__.py
│       │   ├── sources/
│       │   │   ├── __init__.py
│       │   │   ├── base.py        # DataSource ABC
│       │   │   ├── csv_source.py
│       │   │   └── parquet_source.py
│       │   └── ingestion.py       # Data ingestion service
│       ├── features/
│       │   ├── __init__.py
│       │   ├── pipeline.py        # FeaturePipeline ABC
│       │   ├── technical.py       # Technical indicators
│       │   └── rolling.py         # Rolling statistics
│       ├── strategies/
│       │   ├── __init__.py
│       │   ├── base.py            # Strategy ABC
│       │   ├── registry.py        # Strategy plugin registry
│       │   ├── momentum.py        # Example: Momentum strategy
│       │   └── ml_strategy.py     # ML-based strategy base
│       ├── execution/
│       │   ├── __init__.py
│       │   ├── engine.py          # Paper trading engine
│       │   └── portfolio.py       # Portfolio simulator
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── engine.py          # Metrics calculation
│       │   └── export.py          # InfluxDB/Prometheus export
│       └── parallel/
│           ├── __init__.py
│           └── runner.py          # Ray-based parallel runner
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── config/
│   └── config.yaml
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── grafana/
│   └── dashboards/
├── docker-compose.yml
├── pyproject.toml
├── PROMPT.md
├── fix_plan.md
└── README.md
```

---

## 10. Dependencies

### Core
- Python >= 3.10
- pydantic >= 2.0
- polars >= 0.20
- pandas >= 2.0
- numpy >= 1.24
- pyyaml >= 6.0

### ML
- torch >= 2.0
- transformers >= 4.30

### Parallel
- ray >= 2.9

### Metrics & Storage
- influxdb-client >= 1.40
- prometheus-client >= 0.19

### Development
- pytest >= 7.0
- pytest-cov >= 4.0
- black >= 23.0
- ruff >= 0.1
- mypy >= 1.0
