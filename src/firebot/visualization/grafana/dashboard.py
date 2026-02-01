"""Grafana dashboard generator for FireBot metrics.

Generates Grafana-compatible JSON dashboard definitions that can be
imported directly into Grafana or used with provisioning.
"""

from __future__ import annotations

import json
from typing import Any


def _make_panel(
    title: str,
    panel_type: str,
    grid_pos: dict[str, int],
    targets: list[dict[str, Any]],
    panel_id: int,
    unit: str = "",
    description: str = "",
    thresholds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a single Grafana panel definition."""
    panel: dict[str, Any] = {
        "id": panel_id,
        "type": panel_type,
        "title": title,
        "description": description,
        "gridPos": grid_pos,
        "targets": targets,
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
    }
    if unit:
        panel["fieldConfig"] = {
            "defaults": {"unit": unit},
            "overrides": [],
        }
    if thresholds:
        panel.setdefault("fieldConfig", {"defaults": {}, "overrides": []})
        panel["fieldConfig"]["defaults"]["thresholds"] = {
            "mode": "absolute",
            "steps": thresholds,
        }
    return panel


def _prometheus_target(expr: str, legend: str = "") -> dict[str, Any]:
    """Create a Prometheus query target."""
    return {
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
        "expr": expr,
        "legendFormat": legend or "{{strategy_id}}",
        "refId": "A",
    }


def generate_dashboard(
    title: str = "FireBot Trading Dashboard",
    strategy_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a complete Grafana dashboard JSON.

    Args:
        title: Dashboard title
        strategy_ids: Optional list of strategy IDs to filter.
            If None, uses Grafana template variable for all strategies.

    Returns:
        Dict representing a complete Grafana dashboard JSON
    """
    panels: list[dict[str, Any]] = []
    panel_id = 1

    # Row 1: Portfolio Overview (4 stat panels)
    panels.append(_make_panel(
        title="Portfolio Value",
        panel_type="stat",
        grid_pos={"h": 4, "w": 6, "x": 0, "y": 0},
        targets=[_prometheus_target('firebot_portfolio_value{strategy_id=~"$strategy"}')],
        panel_id=panel_id,
        unit="currencyUSD",
        description="Current total portfolio value",
    ))
    panel_id += 1

    panels.append(_make_panel(
        title="Cash Balance",
        panel_type="stat",
        grid_pos={"h": 4, "w": 6, "x": 6, "y": 0},
        targets=[_prometheus_target('firebot_portfolio_cash{strategy_id=~"$strategy"}')],
        panel_id=panel_id,
        unit="currencyUSD",
        description="Available cash",
    ))
    panel_id += 1

    panels.append(_make_panel(
        title="Drawdown",
        panel_type="gauge",
        grid_pos={"h": 4, "w": 6, "x": 12, "y": 0},
        targets=[_prometheus_target('firebot_portfolio_drawdown{strategy_id=~"$strategy"}')],
        panel_id=panel_id,
        unit="percentunit",
        description="Current drawdown from high water mark",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 0.05},
            {"color": "red", "value": 0.10},
        ],
    ))
    panel_id += 1

    panels.append(_make_panel(
        title="High Water Mark",
        panel_type="stat",
        grid_pos={"h": 4, "w": 6, "x": 18, "y": 0},
        targets=[_prometheus_target('firebot_portfolio_hwm{strategy_id=~"$strategy"}')],
        panel_id=panel_id,
        unit="currencyUSD",
        description="Portfolio high water mark",
    ))
    panel_id += 1

    # Row 2: Equity Curve (time series)
    panels.append(_make_panel(
        title="Equity Curve",
        panel_type="timeseries",
        grid_pos={"h": 8, "w": 24, "x": 0, "y": 4},
        targets=[_prometheus_target('firebot_portfolio_value{strategy_id=~"$strategy"}')],
        panel_id=panel_id,
        unit="currencyUSD",
        description="Portfolio value over time",
    ))
    panel_id += 1

    # Row 3: Drawdown Chart (time series)
    panels.append(_make_panel(
        title="Drawdown Over Time",
        panel_type="timeseries",
        grid_pos={"h": 6, "w": 24, "x": 0, "y": 12},
        targets=[_prometheus_target('firebot_portfolio_drawdown{strategy_id=~"$strategy"}')],
        panel_id=panel_id,
        unit="percentunit",
        description="Drawdown from high water mark over time",
    ))
    panel_id += 1

    # Row 4: Performance Metrics (4 stat panels)
    panels.append(_make_panel(
        title="Sharpe Ratio",
        panel_type="stat",
        grid_pos={"h": 4, "w": 6, "x": 0, "y": 18},
        targets=[_prometheus_target('firebot_sharpe_ratio{strategy_id=~"$strategy"}')],
        panel_id=panel_id,
        description="Risk-adjusted return (annualized)",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 1.0},
            {"color": "green", "value": 2.0},
        ],
    ))
    panel_id += 1

    panels.append(_make_panel(
        title="Sortino Ratio",
        panel_type="stat",
        grid_pos={"h": 4, "w": 6, "x": 6, "y": 18},
        targets=[_prometheus_target('firebot_sortino_ratio{strategy_id=~"$strategy"}')],
        panel_id=panel_id,
        description="Downside risk-adjusted return",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 1.5},
            {"color": "green", "value": 2.5},
        ],
    ))
    panel_id += 1

    panels.append(_make_panel(
        title="Win Rate",
        panel_type="gauge",
        grid_pos={"h": 4, "w": 6, "x": 12, "y": 18},
        targets=[_prometheus_target('firebot_win_rate{strategy_id=~"$strategy"}')],
        panel_id=panel_id,
        unit="percentunit",
        description="Percentage of winning trades",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 0.45},
            {"color": "green", "value": 0.55},
        ],
    ))
    panel_id += 1

    panels.append(_make_panel(
        title="Profit Factor",
        panel_type="stat",
        grid_pos={"h": 4, "w": 6, "x": 18, "y": 18},
        targets=[_prometheus_target('firebot_profit_factor{strategy_id=~"$strategy"}')],
        panel_id=panel_id,
        description="Gross profit / gross loss",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 1.0},
            {"color": "green", "value": 1.5},
        ],
    ))
    panel_id += 1

    # Row 5: Trade Activity
    panels.append(_make_panel(
        title="Total Trades",
        panel_type="timeseries",
        grid_pos={"h": 6, "w": 12, "x": 0, "y": 22},
        targets=[_prometheus_target(
            'rate(firebot_trades_total{strategy_id=~"$strategy"}[5m])',
            legend="{{strategy_id}} - {{side}}",
        )],
        panel_id=panel_id,
        description="Trade rate over time",
    ))
    panel_id += 1

    panels.append(_make_panel(
        title="Trade PnL Distribution",
        panel_type="histogram",
        grid_pos={"h": 6, "w": 12, "x": 12, "y": 22},
        targets=[_prometheus_target(
            'firebot_trade_pnl_bucket{strategy_id=~"$strategy"}',
        )],
        panel_id=panel_id,
        unit="currencyUSD",
        description="Distribution of trade PnL",
    ))
    panel_id += 1

    # Template variable for strategy selection
    templating = {
        "list": [
            {
                "name": "strategy",
                "type": "query",
                "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                "query": 'label_values(firebot_portfolio_value, strategy_id)',
                "includeAll": True,
                "multi": True,
                "current": {"text": "All", "value": "$__all"},
                "refresh": 2,
            }
        ]
    }

    # Override if specific strategy_ids provided
    if strategy_ids:
        templating["list"][0]["type"] = "custom"
        templating["list"][0]["query"] = ",".join(strategy_ids)
        del templating["list"][0]["datasource"]

    dashboard: dict[str, Any] = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "panels": panels,
        "schemaVersion": 39,
        "tags": ["firebot", "trading"],
        "templating": templating,
        "time": {"from": "now-24h", "to": "now"},
        "timepicker": {},
        "timezone": "utc",
        "title": title,
        "uid": "firebot-main",
        "version": 1,
        "__inputs": [
            {
                "name": "DS_PROMETHEUS",
                "label": "Prometheus",
                "description": "Prometheus data source for FireBot metrics",
                "type": "datasource",
                "pluginId": "prometheus",
                "pluginName": "Prometheus",
            }
        ],
    }

    return dashboard


def export_dashboard_json(
    filepath: str,
    title: str = "FireBot Trading Dashboard",
    strategy_ids: list[str] | None = None,
) -> None:
    """Export dashboard as a JSON file for Grafana import.

    Args:
        filepath: Output file path
        title: Dashboard title
        strategy_ids: Optional strategy filter
    """
    dashboard = generate_dashboard(title=title, strategy_ids=strategy_ids)
    with open(filepath, "w") as f:
        json.dump(dashboard, f, indent=2)
