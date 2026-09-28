from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .demo import seed_demo
from .domain import utc_now
from .ingestion.eia import EIAClient, normalize_henry_hub, normalize_storage
from .ingestion.reports import DEFAULT_EIA_REPORTS, EIAReportClient
from .ingestion.weather import DEFAULT_LOCATIONS, OpenMeteoClient, normalize_weather
from .orchestrator import OutlookOrchestrator
from .rag import MarketIntelligenceRetriever
from .rag_evaluation import evaluate_retriever, load_evaluation_cases
from .repository import MarketRepository
from .rolling_backtest import RollingBacktestService
from .services import ForecastService, MarketService
from .tracking import MLflowTracker


def _add_xgboost_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--xgb-n-estimators", type=int)
    parser.add_argument("--xgb-max-depth", type=int)
    parser.add_argument("--xgb-learning-rate", type=float)
    parser.add_argument("--xgb-subsample", type=float)
    parser.add_argument("--xgb-colsample-bytree", type=float)
    parser.add_argument("--xgb-min-child-weight", type=float)
    parser.add_argument("--xgb-gamma", type=float)
    parser.add_argument("--xgb-reg-alpha", type=float)
    parser.add_argument("--xgb-reg-lambda", type=float)
    parser.add_argument("--xgb-random-state", type=int)


def _xgboost_parameters(args: argparse.Namespace) -> dict[str, int | float]:
    names = (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "random_state",
    )
    return {
        name: value for name in names if (value := getattr(args, f"xgb_{name}", None)) is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Henry Hub commodity intelligence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the deterministic synthetic demo")
    demo.add_argument("--database", default="data/commodity_ai.db")
    demo.add_argument("--horizon", type=int, choices=(1, 5, 20), default=20)
    demo.add_argument("--run-name")
    _add_xgboost_arguments(demo)
    experiments = subparsers.add_parser(
        "run-mlflow-experiments", help="run five controlled local XGBoost experiments"
    )
    experiments.add_argument("--database", default="data/commodity_ai.db")
    experiments.add_argument("--horizon", type=int, choices=(1, 5, 20), default=20)
    experiments.add_argument("--tracking-uri")
    experiments.add_argument("--experiment-name")
    rolling = subparsers.add_parser(
        "rolling-backtest",
        help="evaluate one XGBoost configuration across historical periods",
    )
    rolling.add_argument("--database", default="data/commodity_ai.db")
    rolling.add_argument("--seed-demo", action="store_true")
    rolling.add_argument("--horizon", type=int, choices=(1, 5, 20), required=True)
    rolling.add_argument("--train-window", type=int, default=40)
    rolling.add_argument("--test-window", type=int, default=10)
    rolling.add_argument("--step", type=int)
    rolling.add_argument("--run-name")
    rolling.add_argument("--tracking-uri")
    rolling.add_argument("--experiment-name")
    _add_xgboost_arguments(rolling)
    ingest = subparsers.add_parser("ingest-eia", help="ingest a versioned Henry Hub snapshot")
    ingest.add_argument("--database", default="data/commodity_ai.db")
    ingest.add_argument("--start")
    ingest.add_argument("--end")
    ingest.add_argument("--raw-dir", default="data/raw/eia")
    live = subparsers.add_parser(
        "ingest-live", help="ingest live prices, storage, weather, and reports"
    )
    live.add_argument("--database", default="data/commodity_ai.db")
    live.add_argument("--start", default="2018-01-01")
    live.add_argument("--raw-dir", default="data/raw")
    rag_eval = subparsers.add_parser("evaluate-rag", help="evaluate retrieval against JSONL cases")
    rag_eval.add_argument("--database", default="data/commodity_ai.db")
    rag_eval.add_argument("--dataset", required=True)
    rag_eval.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "demo":
        path = Path(args.database)
        repository = MarketRepository(path)
        as_of = seed_demo(repository)
        orchestrator = OutlookOrchestrator(
            ForecastService(repository, xgboost_params=_xgboost_parameters(args)),
            MarketService(repository),
            MarketIntelligenceRetriever(repository),
        )
        result = orchestrator.outlook(as_of, args.horizon, run_name=args.run_name)
        result["data_notice"] = "All demo market values are synthetic."
        print(json.dumps(result, indent=2))
    elif args.command == "run-mlflow-experiments":
        repository = MarketRepository(args.database)
        as_of = seed_demo(repository)
        tracker = MLflowTracker(
            enabled=True,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
        )
        matrix: tuple[tuple[str, dict[str, int | float]], ...] = (
            ("baseline", {}),
            ("depth-4", {"max_depth": 4}),
            ("depth-8", {"max_depth": 8}),
            ("lr-005", {"learning_rate": 0.05}),
            ("subsample-100", {"subsample": 1.0}),
        )
        completed = []
        for name, parameters in matrix:
            record = ForecastService(
                repository,
                tracker=tracker,
                xgboost_params=parameters,
            ).run_forecast(as_of, [args.horizon], run_name=name)[0]
            completed.append({"run_name": name, "forecast_id": record.forecast_id})
        print(json.dumps({"experiment_runs": completed}, indent=2))
    elif args.command == "rolling-backtest":
        step = args.step if args.step is not None else args.test_window
        if args.train_window < 40:
            parser.error("rolling-backtest: --train-window must be at least 40")
        if args.test_window < 1 or step < 1:
            parser.error("rolling-backtest: window sizes and step must be positive")
        if step < args.test_window:
            parser.error(
                "rolling-backtest: --step must be at least --test-window for disjoint periods"
            )
        backend = os.getenv("COMMODITY_AI_MODEL_BACKEND", "xgboost").lower()
        if backend != "xgboost":
            parser.error("rolling-backtest supports only COMMODITY_AI_MODEL_BACKEND=xgboost")
        repository = MarketRepository(args.database)
        if args.seed_demo:
            seed_demo(repository)
        tracker = MLflowTracker(
            enabled=True,
            tracking_uri=args.tracking_uri,
            experiment_name=(
                args.experiment_name
                or os.getenv("MLFLOW_EXPERIMENT_NAME")
                or "henry-hub-rolling-backtest"
            ),
        )
        try:
            rolling_result = RollingBacktestService(
                repository,
                tracker,
                _xgboost_parameters(args),
            ).run(
                horizon=args.horizon,
                train_window=args.train_window,
                test_window=args.test_window,
                step=step,
                run_name=args.run_name,
            )
        except ValueError as error:
            parser.error(f"rolling-backtest: {error}")
        summary = rolling_result.evaluation.summary
        print(
            json.dumps(
                {
                    "parent_run_id": rolling_result.parent_run_id,
                    "child_run_ids": rolling_result.child_run_ids,
                    "period_count": summary["period_count"],
                    "prediction_count": summary["prediction_count"],
                    "interval_observation_count": summary["interval_observation_count"],
                    "trailing_origins_skipped": summary["trailing_origins_skipped"],
                    "best_period_by_mae": summary["best_period_by_mae"],
                    "worst_period_by_mae": summary["worst_period_by_mae"],
                    "pooled_metrics": summary["pooled_metrics"],
                    "data_notice": (
                        "Results use deterministic synthetic observations."
                        if summary["data_kind"] == "synthetic"
                        else "Results use the database's published market observations."
                    ),
                },
                indent=2,
            )
        )
    elif args.command == "ingest-eia":
        api_key = os.getenv("EIA_API_KEY", "")
        observed_at = utc_now()
        payload = EIAClient(api_key).fetch_henry_hub_daily(args.start, args.end)
        raw_dir = Path(args.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        snapshot = raw_dir / f"henry_hub_{observed_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        snapshot.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows = normalize_henry_hub(payload, observed_at)
        repository = MarketRepository(args.database)
        repository.add_prices(rows)
        print(json.dumps({"rows": len(rows), "raw_snapshot": str(snapshot)}, indent=2))
    elif args.command == "ingest-live":
        observed_at = utc_now()
        raw_root = Path(args.raw_dir)
        repository = MarketRepository(args.database)

        eia = EIAClient(os.getenv("EIA_API_KEY", ""))
        price_payload = eia.fetch_henry_hub_daily(args.start)
        storage_payload = eia.fetch_lower_48_storage_weekly(args.start)
        eia_dir = raw_root / "eia"
        eia_dir.mkdir(parents=True, exist_ok=True)
        stamp = observed_at.strftime("%Y%m%dT%H%M%SZ")
        (eia_dir / f"henry_hub_{stamp}.json").write_text(
            json.dumps(price_payload, indent=2), encoding="utf-8"
        )
        (eia_dir / f"storage_{stamp}.json").write_text(
            json.dumps(storage_payload, indent=2), encoding="utf-8"
        )
        prices = normalize_henry_hub(price_payload, observed_at)
        storage = normalize_storage(storage_payload, observed_at)
        repository.add_prices(prices)
        repository.add_storage(storage)

        weather_client = OpenMeteoClient()
        weather_payloads = [
            (location, weather_client.fetch_forecast(location)) for location in DEFAULT_LOCATIONS
        ]
        weather_dir = raw_root / "weather"
        weather_dir.mkdir(parents=True, exist_ok=True)
        (weather_dir / f"forecast_{stamp}.json").write_text(
            json.dumps(
                [
                    {"location": location.__dict__, "payload": payload}
                    for location, payload in weather_payloads
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        weather = normalize_weather(weather_payloads, observed_at)
        repository.add_weather(weather)

        report_client = EIAReportClient()
        reports = [report_client.fetch(source, observed_at) for source in DEFAULT_EIA_REPORTS]
        reports_dir = raw_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for report in reports:
            (reports_dir / f"{report.report_id}.txt").write_text(report.raw_text, encoding="utf-8")
        repository.add_reports(reports)
        print(
            json.dumps(
                {
                    "prices": len(prices),
                    "storage": len(storage),
                    "weather": len(weather),
                    "reports": len(reports),
                    "observed_at": observed_at.isoformat(),
                },
                indent=2,
            )
        )
    elif args.command == "evaluate-rag":
        repository = MarketRepository(args.database)
        metrics = evaluate_retriever(
            MarketIntelligenceRetriever(repository),
            load_evaluation_cases(args.dataset),
            args.top_k,
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
