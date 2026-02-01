"""Tests for Grafana dashboard generator."""

import json
from pathlib import Path

import pytest

from firebot.visualization.grafana.dashboard import (
    generate_dashboard,
    export_dashboard_json,
)


class TestGrafanaDashboard:
    """Tests for Grafana dashboard generation."""

    def test_dashboard_has_required_keys(self) -> None:
        """Generated dashboard should have all required Grafana keys."""
        dashboard = generate_dashboard()
        assert "panels" in dashboard
        assert "title" in dashboard
        assert "templating" in dashboard
        assert "schemaVersion" in dashboard
        assert "uid" in dashboard
        assert "tags" in dashboard

    def test_dashboard_title(self) -> None:
        """Dashboard should use the provided title."""
        dashboard = generate_dashboard(title="My Custom Dashboard")
        assert dashboard["title"] == "My Custom Dashboard"

    def test_dashboard_has_panels(self) -> None:
        """Dashboard should contain multiple panels."""
        dashboard = generate_dashboard()
        assert len(dashboard["panels"]) >= 10

    def test_panel_types(self) -> None:
        """Dashboard should have different panel types."""
        dashboard = generate_dashboard()
        types = {p["type"] for p in dashboard["panels"]}
        assert "stat" in types
        assert "timeseries" in types
        assert "gauge" in types

    def test_portfolio_value_panel_exists(self) -> None:
        """Dashboard should have a portfolio value panel."""
        dashboard = generate_dashboard()
        titles = [p["title"] for p in dashboard["panels"]]
        assert "Portfolio Value" in titles
        assert "Equity Curve" in titles
        assert "Drawdown Over Time" in titles

    def test_performance_panels_exist(self) -> None:
        """Dashboard should have performance metric panels."""
        dashboard = generate_dashboard()
        titles = [p["title"] for p in dashboard["panels"]]
        assert "Sharpe Ratio" in titles
        assert "Sortino Ratio" in titles
        assert "Win Rate" in titles
        assert "Profit Factor" in titles

    def test_strategy_template_variable(self) -> None:
        """Dashboard should have a strategy template variable."""
        dashboard = generate_dashboard()
        variables = dashboard["templating"]["list"]
        assert len(variables) >= 1
        strategy_var = variables[0]
        assert strategy_var["name"] == "strategy"
        assert strategy_var["includeAll"] is True
        assert strategy_var["multi"] is True

    def test_custom_strategy_ids(self) -> None:
        """Dashboard with specific strategy_ids should use custom variable."""
        dashboard = generate_dashboard(
            strategy_ids=["momentum_1", "mean_revert_1"]
        )
        strategy_var = dashboard["templating"]["list"][0]
        assert strategy_var["type"] == "custom"
        assert "momentum_1" in strategy_var["query"]
        assert "mean_revert_1" in strategy_var["query"]

    def test_panels_have_prometheus_datasource(self) -> None:
        """All panels should reference the Prometheus datasource."""
        dashboard = generate_dashboard()
        for panel in dashboard["panels"]:
            assert panel["datasource"]["type"] == "prometheus"

    def test_drawdown_gauge_has_thresholds(self) -> None:
        """Drawdown gauge should have color thresholds."""
        dashboard = generate_dashboard()
        drawdown_panel = next(
            p for p in dashboard["panels"] if p["title"] == "Drawdown"
        )
        thresholds = drawdown_panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
        assert len(thresholds) == 3  # green, yellow, red

    def test_export_to_file(self, tmp_path: Path) -> None:
        """Dashboard should export as valid JSON file."""
        filepath = tmp_path / "dashboard.json"
        export_dashboard_json(str(filepath))
        assert filepath.exists()

        with open(filepath) as f:
            data = json.load(f)
        assert data["title"] == "FireBot Trading Dashboard"
        assert len(data["panels"]) >= 10

    def test_dashboard_tags(self) -> None:
        """Dashboard should be tagged for discovery."""
        dashboard = generate_dashboard()
        assert "firebot" in dashboard["tags"]
        assert "trading" in dashboard["tags"]

    def test_unique_panel_ids(self) -> None:
        """All panels should have unique IDs."""
        dashboard = generate_dashboard()
        ids = [p["id"] for p in dashboard["panels"]]
        assert len(ids) == len(set(ids))

    def test_dashboard_is_json_serializable(self) -> None:
        """Full dashboard should be JSON-serializable."""
        dashboard = generate_dashboard()
        serialized = json.dumps(dashboard)
        assert len(serialized) > 0
        deserialized = json.loads(serialized)
        assert deserialized["title"] == dashboard["title"]
