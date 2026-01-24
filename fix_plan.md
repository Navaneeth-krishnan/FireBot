# Ralph Fix Plan

## High Priority

### Phase 1: Foundation
- [x] Set up Python project structure with proper packaging and uv as a package manager (pyproject.toml, src layout)
- [x] Create base data models (OHLCV, Signal, Order, Position, Portfolio)
- [x] Implement Market Data Layer with pluggable data source interface
- [x] Add CSV/Parquet data source adapter for historical data
- [x] Create FeaturePipeline base class with transform() method

### Phase 2: Strategy Engine
- [x] Define Strategy base class with on_data() and generate_signal() methods
- [x] Implement strategy plugin registry (discover and load strategies)
- [x] Create Signal data model (direction, confidence, metadata)
- [x] Add YAML configuration loader for strategies
- [x] Build example MomentumStrategy as reference implementation

### Phase 3: Execution Engine
- [x] Implement Paper Trading Engine with order simulation
- [x] Create Portfolio Simulator (positions, cash, PnL tracking)
- [x] Add risk controls (max drawdown, position size limits)
- [x] Implement instant fill execution model
- [x] Add stop-loss / take-profit order types

## Medium Priority

### Phase 4: Parallel Execution
- [x] Set up Ray for parallel strategy execution
- [x] Implement shared data feed distribution
- [x] Create independent virtual portfolio per strategy
- [x] Add strategy isolation and error handling

### Phase 5: Metrics & Storage
- [x] Implement Metrics Engine (Sharpe, Sortino, max drawdown, win rate)
- [x] Add InfluxDB/Prometheus metrics export
- [x] Create time-series storage for trade history
- [x] Implement experiment metadata storage

### Phase 6: Visualization
- [ ] Create Grafana dashboard templates
- [ ] Add equity curve visualization
- [ ] Implement drawdown charts
- [ ] Add strategy comparison views

## Low Priority

### Phase 7: ML Integration
- [ ] Add MLStrategy base class with model loading
- [ ] Implement feature store interface
- [ ] Add model versioning support
- [ ] Create Transformer-based strategy example

### Phase 8: Advanced Features
- [ ] Implement Signal Aggregation Layer (ensemble, voting)
- [ ] Add backtesting framework with deterministic replay
- [ ] Create forward testing mode
- [ ] Add Docker deployment configuration

### Future Enhancements
- [ ] Live market data integration
- [ ] Real trading gateway interface
- [ ] AutoML strategy generation
- [ ] Portfolio optimization engine

## Completed
- [x] Project initialization
- [x] PRD analysis and Ralph conversion

## Notes

### Architecture Decisions
- Use abstract base classes for extensibility
- Prefer composition over inheritance for strategies
- Keep data flow unidirectional: Data -> Features -> Signals -> Orders -> Portfolio
- All components should be independently testable and follow TDD workflow

### Data Model Priorities
1. OHLCV - Core price data
2. Signal - Strategy output
3. Order - Execution intent
4. Position - Current holdings
5. Portfolio - Aggregate state

### Risk Management
- Default max drawdown: 10%
- Default max position size: 5% of portfolio
- Auto-disable strategy on sustained underperformance

### Tech Stack Reminders
- Python 3.10+ required
- Use Polars for large datasets (faster than Pandas)
- Ray for parallelism (single-node focus)
- Pydantic for data validation
