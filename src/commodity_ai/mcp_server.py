from __future__ import annotations

import json
import os

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as error:  # pragma: no cover - optional adapter
    raise RuntimeError("Install the MCP extra: pip install -e '.[mcp]'") from error

from .domain import parse_date, parse_datetime, utc_now
from .rag import MarketIntelligenceRetriever, SearchFilters
from .repository import MarketRepository
from .services import ForecastService, MarketService

repo = MarketRepository(os.getenv("COMMODITY_AI_DB", "data/commodity_ai.db"))
forecast_service = ForecastService(repo)
market_service = MarketService(repo)
retrieval = MarketIntelligenceRetriever(repo)
mcp = MCPServer("energy-market-mcp")


@mcp.tool()
def get_henry_hub_history(start_date: str, end_date: str, as_of: str) -> list[dict[str, object]]:
    """Get price observations available at the supplied point in time."""
    rows = repo.prices_as_of(parse_datetime(as_of), parse_date(start_date), parse_date(end_date))
    return [
        {
            "date": row.observation_date.isoformat(),
            "price": row.price,
            "unit": row.unit,
            "publication_timestamp": row.publication_timestamp.isoformat(),
            "source": row.source,
        }
        for row in rows
    ]


@mcp.tool()
def get_market_snapshot(as_of: str) -> dict[str, object]:
    return market_service.snapshot(parse_datetime(as_of))


@mcp.tool()
def get_storage_summary(as_of: str) -> dict[str, object]:
    snapshot = market_service.snapshot(parse_datetime(as_of))
    return {key: value for key, value in snapshot.items() if "storage" in key or key == "as_of"}


@mcp.tool()
def get_weather_signal(as_of: str, horizon: int) -> dict[str, object]:
    return market_service.weather_signal(parse_datetime(as_of), horizon)


@mcp.tool()
def run_forecast(as_of: str, horizons: list[int]) -> list[dict[str, object]]:
    """Run numerical models. An LLM must not modify the returned forecast values."""
    return [row.to_dict() for row in forecast_service.run_forecast(parse_datetime(as_of), horizons)]


@mcp.tool()
def explain_forecast(forecast_id: str) -> dict[str, object]:
    row = repo.get_forecast(forecast_id)
    if row is None:
        raise ValueError("forecast not found")
    return {
        "forecast_id": row.forecast_id,
        "drivers": [driver.__dict__ for driver in row.drivers],
        "baseline_comparison": row.point_forecast - row.baseline_forecast,
        "historical_model_accuracy": row.metrics,
        "uncertainty": {"p10": row.p10, "p50": row.p50, "p90": row.p90},
    }


@mcp.tool()
def search_market_intelligence(
    query: str,
    as_of: str,
    top_k: int = 5,
    commodity: str | None = None,
    region: str | None = None,
    document_type: str | None = None,
    publisher: str | None = None,
) -> list[dict[str, object]]:
    """Hybrid report-chunk search with hard point-in-time and metadata filters."""
    filters = SearchFilters(commodity, region, document_type, publisher)
    return [
        item.__dict__ for item in retrieval.search(query, parse_datetime(as_of), top_k, filters)
    ]


@mcp.tool()
def evaluate_forecast(forecast_id: str, evaluation_as_of: str) -> dict[str, object]:
    return forecast_service.evaluate_forecast(forecast_id, parse_datetime(evaluation_as_of))


@mcp.resource("market://henry-hub/metadata")
def market_metadata() -> str:
    return (
        '{"commodity":"Henry Hub Natural Gas","unit":"USD/MMBtu",'
        '"horizons":[1,5,20],"numerical_forecasts_generated_by":"ML model"}'
    )


@mcp.resource("model://current")
def current_model() -> str:
    return '{"model_version":"xgboost-direct-0.2","status":"educational"}'


@mcp.resource("market://henry-hub/latest")
def latest_market() -> str:
    return json.dumps(market_service.snapshot(utc_now()))


@mcp.resource("forecast://{forecast_id}")
def forecast_resource(forecast_id: str) -> str:
    forecast = repo.get_forecast(forecast_id)
    if forecast is None:
        raise ValueError("forecast not found")
    return json.dumps(forecast.to_dict())


@mcp.resource("report://{report_id}")
def report_resource(report_id: str) -> str:
    report = repo.get_report(report_id)
    if report is None:
        raise ValueError("report not found")
    return report.raw_text


@mcp.prompt()
def explain_henry_hub_outlook(as_of: str, horizon: int = 20) -> str:
    return (
        f"Explain the {horizon}-business-day Henry Hub outlook as of {as_of}. "
        "Call run_forecast, explain_forecast, get_market_snapshot, and "
        "search_market_intelligence. Separate quantitative drivers from report evidence, "
        "identify supporting and contradicting evidence, reproduce model numbers exactly, "
        "and include citations, uncertainty, and historical performance."
    )


if __name__ == "__main__":
    mcp.run()
