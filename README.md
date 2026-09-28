# Henry Hub Commodity Intelligence

A point-in-time-correct forecasting and market-intelligence platform for Henry Hub natural gas. Numerical forecasts are produced by XGBoost; the LLM can retrieve and explain them but cannot create or modify them.

## Implemented

- Live EIA Henry Hub price and Lower-48 storage ingestion, population-weighted Open-Meteo weather, and EIA report ingestion with retained raw snapshots
- Leakage-safe features and direct XGBoost models for 1, 5, and 20 observed-business-day horizons
- Persistence and five-day-average baselines, walk-forward validation, out-of-sample residual calibration, interval coverage, pinball loss, and skill metrics
- Native XGBoost TreeSHAP contributions persisted with each forecast
- Optional MLflow experiment, metric, dataset fingerprint, Git revision, feature manifest, and model artifact tracking
- Fixed-width rolling XGBoost backtests with leakage-safe prior-only interval calibration and nested MLflow period runs
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

The Forecast view offers two explicit training modes. **Historical (bulk EIA,
non-vintage)** trains immediately on the latest bulk price and storage history, keeps
the normal storage-release lag, and omits unavailable historical weather vintages.
Its evaluation can be revision-biased and is labeled accordingly. **Point-in-time**
replays only data visible at each cutoff and therefore requires accumulated ingestion
snapshots before it can train.

## RAG and agent configuration

The default `hash` embedding provider is local and deterministic. To use hosted semantic embeddings:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:COMMODITY_AI_EMBEDDING_PROVIDER = "openai"
$env:OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

The Ask AI view also requires `OPENAI_API_KEY`. `OPENAI_MODEL` defaults to `gpt-5.6-sol`. The agent connects to `COMMODITY_AI_MCP_URL` when set; otherwise it uses the same MCP server in-process.

Run the included retrieval benchmark with:

```powershell
commodity-ai evaluate-rag --database data/commodity_ai.db --dataset data/rag_evaluation.jsonl
```

## MLflow

MLflow is included in the `ml` and `all` extras. It can also be installed directly with
`python -m pip install mlflow`. Start the local tracking server from the repository root:

```powershell
$projectRoot = (Resolve-Path .).Path.Replace("\", "/")
.\.venv\Scripts\python.exe -m mlflow server `
  --backend-store-uri "sqlite:///$projectRoot/data/mlflow.db" `
  --default-artifact-root "$projectRoot/data/mlartifacts" `
  --port 5000
```

In a second terminal, enable tracking and run the existing forecasting demo:

```powershell
$env:COMMODITY_AI_MLFLOW_ENABLED = "true"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_EXPERIMENT_NAME = "henry-hub-xgboost"
commodity-ai demo --database data/commodity_ai.db --horizon 20 --run-name baseline
```

Open `http://localhost:5000` and select `henry-hub-xgboost`. Each trained horizon is one
run containing XGBoost hyperparameters, feature and period metadata, `test_mae`, `test_rmse`,
`test_mape`, backtest predictions, feature configuration, feature importance, and the trained
model. MAPE is stored as a fraction and excludes observations whose actual value is zero.
Prediction intervals remain calibrated only from walk-forward out-of-sample residuals.

Run the controlled five-run comparison matrix against the same synthetic data and period with:

```powershell
commodity-ai run-mlflow-experiments --database data/commodity_ai.db --horizon 20
```

The matrix records `baseline`, `depth-4`, `depth-8`, `lr-005`, and `subsample-100`. For an
individual run, the `demo` command also accepts XGBoost overrides such as
`--xgb-max-depth 4`, `--xgb-learning-rate 0.05`, and `--xgb-subsample 1.0`. When multiple
horizons are requested programmatically, the horizon is appended to the supplied run name.
Run names are labels only; MLflow run IDs remain authoritative. Tracking failures are raised
explicitly and are not silently discarded.

Evaluate one XGBoost configuration over successive, disjoint historical periods with:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_EXPERIMENT_NAME = "henry-hub-rolling-backtest"
commodity-ai rolling-backtest --database data/commodity_ai.db --seed-demo `
  --horizon 20 --train-window 40 --test-window 10 --step 10 `
  --run-name xgb-20d-rolling
```

This command always enables MLflow tracking. It creates one `rolling_summary` parent run and
one nested `rolling_window` child run for each complete period. Each child contains its own
metrics, auditable training/evaluation snapshots and fingerprints, predictions, configuration,
and fitted XGBoost model. Interval bounds at an origin use only earlier out-of-sample residuals
whose targets were already published by that origin's cutoff.

The deterministic demo currently produces five complete periods (50 predictions) and reports
two trailing origins as skipped rather than evaluating an incomplete sixth period. These are
the results from an actual run with the command above; interval coverage has 10 eligible
prior-calibrated observations in every period:

| Period | Test dates | N | XGBoost MAE | Persistence MAE | Skill vs. persistence | P10-P90 coverage |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 2025-06-03 to 2025-06-16 | 10 | 0.569 | 0.346 | -0.645 | 0.00 |
| 1 | 2025-06-17 to 2025-06-30 | 10 | 0.586 | 0.169 | -2.460 | 0.20 |
| 2 (best) | 2025-07-01 to 2025-07-14 | 10 | 0.269 | 0.315 | 0.145 | 1.00 |
| 3 | 2025-07-15 to 2025-07-28 | 10 | 0.564 | 0.659 | 0.144 | 0.40 |
| 4 (worst) | 2025-07-29 to 2025-08-11 | 10 | 0.768 | 0.550 | -0.396 | 0.00 |

The pooled MAE is 0.551 versus 0.408 for persistence (skill -0.352). This synthetic result is a
workflow demonstration only; it is not evidence of live-market predictive performance.

In the UI, select multiple runs to compare their parameters, metrics, and artifacts:

```text
Experiment
└── Runs
    ├── Parameters and train/test periods
    ├── Metrics (MAE, RMSE, MAPE)
    ├── Configuration and evaluation artifacts
    └── Trained XGBoost model
```

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
