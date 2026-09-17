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
from .services import ForecastService, MarketService


def main() -> None:
    parser = argparse.ArgumentParser(description="Henry Hub commodity intelligence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the deterministic synthetic demo")
    demo.add_argument("--database", default="data/commodity_ai.db")
    demo.add_argument("--horizon", type=int, choices=(1, 5, 20), default=20)
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
            ForecastService(repository),
            MarketService(repository),
            MarketIntelligenceRetriever(repository),
        )
        result = orchestrator.outlook(as_of, args.horizon)
        result["data_notice"] = "All demo market values are synthetic."
        print(json.dumps(result, indent=2))
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
