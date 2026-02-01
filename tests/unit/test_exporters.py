"""Tests for Prometheus exporter and trade store."""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from firebot.metrics.exporters import PrometheusExporter
from firebot.metrics.trade_store import TradeStore, TradeEvent


class TestPrometheusExporter:
    """Tests for Prometheus metrics exporter."""

    def test_exporter_initialization(self) -> None:
        """Exporter should initialize with strategy_id."""
        exporter = PrometheusExporter(strategy_id="test_strat")
        assert exporter.strategy_id == "test_strat"

    def test_update_portfolio_metrics(self) -> None:
        """Exporter should update portfolio gauges."""
        exporter = PrometheusExporter(strategy_id="test_strat")
        exporter.update_portfolio(
            cash=Decimal("85000"),
            total_value=Decimal("102000"),
            drawdown=Decimal("0.02"),
            high_water_mark=Decimal("104000"),
        )
        output = exporter.generate_metrics().decode("utf-8")
        assert "firebot_portfolio_value" in output
        assert "102000" in output
        assert "firebot_portfolio_cash" in output
        assert "firebot_portfolio_drawdown" in output

    def test_update_performance_metrics(self) -> None:
        """Exporter should update performance gauges."""
        exporter = PrometheusExporter(strategy_id="test_strat")
        exporter.update_performance(
            sharpe=1.5,
            sortino=2.1,
            win_rate_val=0.6,
            profit_factor_val=1.8,
        )
        output = exporter.generate_metrics().decode("utf-8")
        assert "firebot_sharpe_ratio" in output
        assert "firebot_sortino_ratio" in output
        assert "firebot_win_rate" in output
        assert "firebot_profit_factor" in output

    def test_record_trade(self) -> None:
        """Exporter should record trade counters and histogram."""
        exporter = PrometheusExporter(strategy_id="test_strat")
        exporter.record_trade(pnl=500.0, side="buy")
        exporter.record_trade(pnl=-200.0, side="sell")
        output = exporter.generate_metrics().decode("utf-8")
        assert "firebot_trades_total" in output
        assert "firebot_trade_pnl" in output

    def test_generate_prometheus_format(self) -> None:
        """Output should be valid Prometheus exposition format."""
        exporter = PrometheusExporter(strategy_id="test_strat")
        exporter.update_portfolio(
            cash=Decimal("100000"),
            total_value=Decimal("100000"),
        )
        output = exporter.generate_metrics()
        assert isinstance(output, bytes)
        text = output.decode("utf-8")
        # Prometheus format has HELP and TYPE lines
        assert "# HELP" in text
        assert "# TYPE" in text

    def test_strategy_label_in_output(self) -> None:
        """Metrics should include strategy_id label."""
        exporter = PrometheusExporter(strategy_id="momentum_1")
        exporter.update_portfolio(
            cash=Decimal("100000"),
            total_value=Decimal("100000"),
        )
        output = exporter.generate_metrics().decode("utf-8")
        assert 'strategy_id="momentum_1"' in output


class TestTradeStore:
    """Tests for trade history storage."""

    def test_store_initialization(self) -> None:
        """Store should initialize empty."""
        store = TradeStore()
        assert store.count == 0

    def test_record_trade(self) -> None:
        """Store should record a trade event."""
        store = TradeStore()
        event = store.record(
            strategy_id="test_strat",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            entry_price=Decimal("150"),
            exit_price=Decimal("160"),
            pnl=Decimal("1000"),
        )
        assert store.count == 1
        assert isinstance(event, TradeEvent)
        assert event.symbol == "AAPL"
        assert event.pnl == "1000"

    def test_query_by_strategy(self) -> None:
        """Store should filter trades by strategy_id."""
        store = TradeStore()
        store.record("strat_1", "AAPL", "buy", Decimal("100"), Decimal("150"), Decimal("160"), Decimal("1000"))
        store.record("strat_2", "MSFT", "buy", Decimal("50"), Decimal("300"), Decimal("310"), Decimal("500"))
        store.record("strat_1", "GOOG", "sell", Decimal("20"), Decimal("2800"), Decimal("2700"), Decimal("-2000"))

        result = store.query(strategy_id="strat_1")
        assert len(result) == 2
        assert all(t.strategy_id == "strat_1" for t in result)

    def test_query_by_symbol(self) -> None:
        """Store should filter trades by symbol."""
        store = TradeStore()
        store.record("strat_1", "AAPL", "buy", Decimal("100"), Decimal("150"), Decimal("160"), Decimal("1000"))
        store.record("strat_1", "MSFT", "buy", Decimal("50"), Decimal("300"), Decimal("310"), Decimal("500"))

        result = store.query(symbol="AAPL")
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    def test_query_by_time_range(self) -> None:
        """Store should filter trades by time range."""
        store = TradeStore()
        now = datetime.now(timezone.utc)

        store.record("strat_1", "AAPL", "buy", Decimal("100"), Decimal("150"), Decimal("160"), Decimal("1000"), timestamp=now - timedelta(hours=2))
        store.record("strat_1", "AAPL", "sell", Decimal("100"), Decimal("160"), Decimal("155"), Decimal("-500"), timestamp=now - timedelta(hours=1))
        store.record("strat_1", "AAPL", "buy", Decimal("100"), Decimal("155"), Decimal("165"), Decimal("1000"), timestamp=now)

        result = store.query(start=now - timedelta(hours=1, minutes=30))
        assert len(result) == 2  # Last two trades

    def test_get_strategies(self) -> None:
        """Store should return unique strategy IDs."""
        store = TradeStore()
        store.record("strat_1", "AAPL", "buy", Decimal("100"), Decimal("150"), Decimal("160"), Decimal("1000"))
        store.record("strat_2", "MSFT", "buy", Decimal("50"), Decimal("300"), Decimal("310"), Decimal("500"))
        store.record("strat_1", "GOOG", "sell", Decimal("20"), Decimal("2800"), Decimal("2700"), Decimal("-2000"))

        strategies = store.get_strategies()
        assert set(strategies) == {"strat_1", "strat_2"}

    def test_persist_and_load(self, tmp_path: Path) -> None:
        """Store should persist trades to disk and reload them."""
        filepath = tmp_path / "trades.jsonl"

        # Write trades
        store1 = TradeStore(persist_path=filepath)
        store1.record("strat_1", "AAPL", "buy", Decimal("100"), Decimal("150"), Decimal("160"), Decimal("1000"))
        store1.record("strat_1", "MSFT", "sell", Decimal("50"), Decimal("300"), Decimal("290"), Decimal("-500"))
        assert store1.count == 2

        # Reload from disk
        store2 = TradeStore(persist_path=filepath)
        assert store2.count == 2
        assert store2.query(symbol="AAPL")[0].pnl == "1000"
