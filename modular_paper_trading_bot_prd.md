# Product Requirements Document (PRD)
## Project: Modular ML-Based Paper Trading Bot

## 1. Overview
This project aims to build a **modular, Python-first paper trading platform** that allows a machine learning engineer to:
- Plug in/out multiple trading strategies
- Run strategies in parallel
- Simulate trades using real market data (paper trading)
- Evaluate performance via metrics and dashboards
- Iterate quickly on ML/quant ideas

The system is research-focused, not production trading-focused, and is designed for **experimentation, learning, and strategy validation** rather than financial execution.

---

## 2. Goals
### Primary Goals
- Modular strategy framework (plug-and-play strategies)
- Parallel strategy execution
- Real-time + historical paper trading
- Clean separation of concerns (data, strategies, execution, evaluation)
- Python-native implementation
- Performance visualization via Grafana

### Secondary Goals
- Strategy benchmarking
- Reproducible experiments
- Backtesting + forward testing
- ML-friendly pipeline integration
- Research-grade logging and metrics

### Non-Goals (for now)
- Real-money trading
- Regulatory compliance
- Ultra-low latency execution
- Exchange co-location

---

## 3. Target User
**Primary user**: You (ML Engineer)
- Strong Python skills
- Comfortable with ML pipelines
- Limited finance background
- Research-oriented mindset
- Interested in experimentation, not high-frequency trading

---

## 4. System Architecture (High Level)

```
Market Data Layer
   ↓
Data Ingestion Service
   ↓
Feature Engineering Layer
   ↓
Strategy Engine (Plugin System)
   ↓
Signal Aggregation Layer
   ↓
Paper Trading Engine
   ↓
Portfolio Simulator
   ↓
Metrics Engine
   ↓
Time Series DB (Prometheus/InfluxDB)
   ↓
Grafana Dashboard
```

---

## 5. Core Components

### 5.1 Market Data Layer
**Responsibilities:**
- Fetch historical (initially) and live (future) market data
- Support multi-asset correlation analysis across markets

**Market Scope (Updated):**
- Primary focus: **Stocks**
- Secondary (future): Cross-market tracking for correlation analysis

**Data Frequency:**
- Tick to **1-hour resolution** (configurable)

**Data Types:**
- OHLCV
- Corporate actions (splits, dividends)
- Market-wide indices (for regime/context features)

**Sources (pluggable):**
- Stock market APIs
- Historical CSV / Parquet datasets

---

### 5.2 Data Ingestion Service
- Normalizes data formats
- Handles:
  - Streaming ingestion
  - Batch ingestion
- Time-indexed storage

---

### 5.3 Feature Engineering Layer
- Technical indicators
- Rolling statistics
- ML features
- Custom feature pipelines

Design:
```python
class FeaturePipeline:
    def transform(self, raw_data) -> features
```

---

### 5.4 Strategy Engine (Plugin Architecture)

**Primary Focus:** ML-heavy strategies

**Core abstraction:**
```python
class Strategy:
    def on_data(self, data):
        pass
    
    def generate_signal(self, features) -> Signal:
        pass
```

**Strategy Types:**
- Transformer-based time series models (primary)
- ML regression / classification (secondary)
- Hybrid ML + rule-based

**Transformer Use Cases:**
- Price movement forecasting
- Volatility prediction
- Cross-asset correlation modeling
- Regime-aware signal generation

---

### 5.5 Parallel Strategy Runner
- Run multiple strategies simultaneously
- Independent virtual portfolios
- Shared data feed

Modes:
- Single-asset multi-strategy
- Multi-asset multi-strategy

---

### 5.6 Signal Aggregation Layer (Optional)
Allows:
- Ensemble strategies
- Voting systems
- Confidence-weighted aggregation
- Risk-adjusted blending

---

### 5.7 Paper Trading Engine
Simulates trading at an **abstract level** for research speed.

Simulates:
- Order placement
- Instant fills (default)
- Optional configurable stop-loss / take-profit

**Risk Controls:**
- Strategy auto-disable on:
  - Max drawdown breach
  - Sustained underperformance
- Manual configuration of stop-loss per strategy

---

### 5.8 Portfolio Simulator
Tracks:
- Positions
- Cash
- PnL
- Drawdown
- Exposure
- Risk metrics

---

### 5.9 Metrics Engine
Metrics:
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Win rate
- Profit factor
- Volatility
- Return distribution

---

### 5.10 Storage Layer
- Time series DB for metrics
- Object storage for models
- Experiment metadata DB

---

### 5.11 Visualization Layer
**Grafana Dashboards:**
- Strategy performance
- Equity curves
- Drawdowns
- Trade distribution
- Correlation between strategies
- Risk exposure

---

## 6. Configuration System

YAML/JSON based:
```yaml
strategies:
  - name: momentum_v1
    class: MomentumStrategy
    params:
      window: 20
  - name: ml_v1
    class: MLStrategy
    model_path: models/model.pkl

assets:
  - BTCUSDT
  - ETHUSDT

risk:
  max_position_size: 5%
  max_drawdown: 10%
```

---

## 7. Experimentation Workflow

1. Define strategy
2. Register strategy plugin
3. Configure in YAML
4. Run backtest
5. Evaluate metrics
6. Run paper trading
7. Monitor Grafana
8. Iterate

---

## 8. ML Integration

Support for:
- Feature stores
- Model versioning
- Offline training
- Online inference
- Model comparison

Example:
```python
class MLStrategy(Strategy):
    def __init__(self, model):
        self.model = model
```

---

## 9. Engineering Principles

- Modular
- Testable
- Deterministic backtests
- Reproducible experiments
- Versioned strategies
- Clear interfaces
- Data-first design

---

## 10. Future Extensions

- Real trading gateway
- AutoML strategy generation
- Genetic strategy evolution
- Meta-learning
- Strategy marketplaces
- Multi-agent RL
- Portfolio optimization engine

---

## 11. Tech Stack (Proposed)

- Python (core logic)
- PyTorch (Transformer models)
- Pandas / Polars (data handling)
- FastAPI (optional services)
- Ray (parallel strategy execution, single-node)
- InfluxDB or Prometheus (metrics)
- Grafana (dashboards)
- Docker (local deployment)

---

## 12. Key Design Philosophy

> "Treat strategies as models, markets as datasets, and trading as inference."

Additional principles:
- Research speed > execution realism
- ML observability over PnL obsession
- Correlation understanding over point prediction
- Single-node first, scalable later

---

