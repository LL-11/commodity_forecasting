# Henry Hub Commodity Intelligence

A point-in-time-correct forecasting and market-intelligence platform for Henry Hub natural gas. Numerical forecasts are produced by XGBoost; the LLM can retrieve and explain them but cannot create or modify them.

## Implemented

- Live EIA Henry Hub price and Lower-48 storage ingestion, population-weighted Open-Meteo weather, and EIA report ingestion with retained raw snapshots
- Leakage-safe features and direct XGBoost models for 1, 5, and 20 observed-business-day horizons
- Persistence and five-day-average baselines, walk-forward validation, out-of-sample residual calibration, interval coverage, pinball loss, and skill metrics
- Native XGBoost TreeSHAP contributions persisted with each forecast
- Optional MLflow experiment, metric, dataset fingerprint, Git revision, feature manifest, and model artifact tracking
- Point-in-time hybrid BM25/TF-IDF/vector retrieval, persistent embeddings, deterministic reranking, metadata filters, citations, and a RAG benchmark CLI
- MCP tools, resources, templates, and an OpenAI Responses API agent that discovers and calls the MCP server
- FastAPI endpoints and a seven-view Streamlit application for market, forecast, drivers, intelligence, Ask AI, evaluation, and data quality
- Docker Compose services for Streamlit, FastAPI, MCP over streamable HTTP, and MLflow

The repository ships with deterministic synthetic data for validation. Demo observations are never labeled as live data.

## Local quick start

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[all,dev]"
commodity-ai demo --database data/commodity_ai.db
streamlit run app/streamlit_app.py
```

The dashboard is available at `http://localhost:8501`. Start the other local adapters when needed:

```powershell
uvicorn commodity_ai.api:app --reload
python -m commodity_ai.mcp_server
mlflow server --backend-store-uri sqlite:///data/mlflow.db --port 5000
```

## Live ingestion

Create an EIA Open Data API key, then run the combined ingestion job:

```powershell
$env:EIA_API_KEY = "your-key"
commodity-ai ingest-live --start 2018-01-01 --database data/commodity_ai.db
```

This writes immutable source responses under `data/raw/` and normalized price, storage, weather, report, and report-chunk rows to SQLite. Because EIA series responses do not carry row-level release times, first-seen ingestion time is used conservatively as publication time.

## RAG and agent configuration

The default `hash` embedding provider is local and deterministic. To use hosted semantic embeddings:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:COMMODITY_AI_EMBEDDING_PROVIDER = "openai"
$env:OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

The Ask AI view also requires `OPENAI_API_KEY`. `OPENAI_MODEL` defaults to `gpt-5.5`. The agent connects to `COMMODITY_AI_MCP_URL` when set; otherwise it uses the same MCP server in-process.

Run the included retrieval benchmark with:

```powershell
commodity-ai evaluate-rag --database data/commodity_ai.db --dataset data/rag_evaluation.jsonl
```

## MLflow

Tracking is opt-in for local runs:

```powershell
$env:COMMODITY_AI_MLFLOW_ENABLED = "true"
$env:MLFLOW_TRACKING_URI = "sqlite:///data/mlflow.db"
```

Every horizon is logged as a separate run. Prediction intervals are calibrated only from walk-forward out-of-sample residuals, not in-sample fit residuals.

## Containers

```powershell
docker compose up --build
```

- Dashboard: `http://localhost:8501`
- API and OpenAPI: `http://localhost:8000/docs`
- MCP streamable HTTP: `http://localhost:8001/mcp`
- MLflow: `http://localhost:5000`

The services share `./data`. Put `EIA_API_KEY`, `OPENAI_API_KEY`, and optionally `OPENAI_MODEL` in a local `.env` file before starting Compose.

## Verification

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src app
docker compose config
```

## Point-in-time invariant

Every source record carries observation, publication, and ingestion timestamps. All feature and retrieval queries enforce `publication_timestamp <= as_of`. Forecast records retain model version, forecast timestamp, latest-data timestamp, calibrated interval, drivers, and historical metrics.

```text
structured data -> point-in-time features -> XGBoost -> forecast + TreeSHAP
reports -> chunks -> embeddings + lexical retrieval -> reranker -> cited evidence
forecast + evidence -> MCP tools -> LLM agent -> grounded explanation
```

## Deliberate remaining scope

The specification's post-MVP items are not disguised as complete: SARIMAX comparison, expanded fundamentals, regime-segmented evaluation, a 50–100-question frozen RAG/agent evaluation corpus, answer-level faithfulness judging, cloud deployment, and later commodities remain future work. This software is educational and is not trading advice.
