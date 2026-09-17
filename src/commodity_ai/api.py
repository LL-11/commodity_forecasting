from __future__ import annotations

import os

try:
    from fastapi import FastAPI, HTTPException
except ImportError as error:  # pragma: no cover - optional adapter
    raise RuntimeError("Install the API extra: pip install -e '.[api]'") from error

from .domain import parse_datetime
from .rag import MarketIntelligenceRetriever, SearchFilters
from .repository import MarketRepository
from .services import ForecastService, MarketService

repository = MarketRepository(os.getenv("COMMODITY_AI_DB", "data/commodity_ai.db"))
forecasts = ForecastService(repository)
market = MarketService(repository)
retriever = MarketIntelligenceRetriever(repository)
app = FastAPI(title="Henry Hub Commodity Intelligence", version="0.2.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/market/snapshot")
def market_snapshot(as_of: str) -> dict[str, object]:
    try:
        return market.snapshot(parse_datetime(as_of))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/forecasts")
def run_forecast(as_of: str, horizons: str = "1,5,20") -> list[dict[str, object]]:
    try:
        requested = [int(value) for value in horizons.split(",")]
        return [item.to_dict() for item in forecasts.run_forecast(parse_datetime(as_of), requested)]
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/intelligence/search")
def search(
    query: str,
    as_of: str,
    top_k: int = 5,
    commodity: str | None = None,
    region: str | None = None,
    document_type: str | None = None,
    publisher: str | None = None,
) -> list[dict[str, object]]:
    try:
        filters = SearchFilters(commodity, region, document_type, publisher)
        return [
            item.__dict__ for item in retriever.search(query, parse_datetime(as_of), top_k, filters)
        ]
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/agent/ask")
def ask_agent(question: str, as_of: str) -> dict[str, object]:
    try:
        from .agent import MCPResponsesAgent

        return (
            MCPResponsesAgent().answer_sync(question, parse_datetime(as_of).isoformat()).to_dict()
        )
    except (ValueError, RuntimeError) as error:
        raise HTTPException(400, str(error)) from error


@app.get("/evaluation/forecasts")
def forecast_evaluations(limit: int = 200) -> list[dict[str, object]]:
    return repository.forecast_evaluations(max(1, min(limit, 1000)))


@app.get("/data-quality")
def data_quality() -> list[dict[str, object]]:
    return repository.data_quality_summary()
