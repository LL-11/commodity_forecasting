# Contributing

Contributions are welcome through GitHub issues and pull requests.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[all,dev]"
```

Before submitting a pull request, run:

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src app
docker compose config --quiet
```

Never commit API keys, `.env` files, databases, raw source snapshots, MLflow
artifacts, or proprietary market data. New data sources must document their terms,
attribution requirements, and point-in-time availability assumptions.

Forecasting changes should include tests demonstrating whether the result is
point-in-time safe or intentionally non-vintage. Do not present backtests based on
revised bulk data as genuine point-in-time performance.
