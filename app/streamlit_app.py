from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import streamlit as st

from commodity_ai.dashboard import build_henry_hub_price_chart, prepare_price_chart_data
from commodity_ai.domain import ForecastRecord, parse_datetime
from commodity_ai.rag import MarketIntelligenceRetriever, SearchFilters
from commodity_ai.rag_evaluation import evaluate_retriever, load_evaluation_cases
from commodity_ai.repository import MarketRepository
from commodity_ai.services import ForecastService, MarketService

st.set_page_config(page_title="Henry Hub Intelligence", page_icon="🔥", layout="wide")
st.title("Henry Hub Commodity Intelligence")
st.caption(
    "Point-in-time XGBoost forecasts with calibrated intervals, TreeSHAP drivers, "
    "hybrid retrieval, and an MCP-grounded analyst."
)


@st.cache_resource
def services() -> tuple[
    MarketRepository, MarketService, ForecastService, MarketIntelligenceRetriever
]:
    repository = MarketRepository(os.getenv("COMMODITY_AI_DB", "data/commodity_ai.db"))
    return (
        repository,
        MarketService(repository),
        ForecastService(repository),
        MarketIntelligenceRetriever(repository),
    )


repository, market, forecasts, retriever = services()


def number(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"expected a numeric database value, got {type(value).__name__}")
    return float(value)


quality = repository.data_quality_summary()
price_quality = next((row for row in quality if row["dataset"] == "prices"), None)
if price_quality is None:
    default_as_of = datetime.now(UTC).isoformat()
    st.sidebar.warning(
        "No price data is available yet. Load or seed data before running forecast queries."
    )
else:
    default_as_of = str(price_quality["latest_timestamp"] or datetime.now(UTC).isoformat())
as_of_text = st.sidebar.text_input("As-of timestamp (UTC)", default_as_of)
st.sidebar.caption("Every query excludes information published after this timestamp.")
as_of = None
try:
    as_of = parse_datetime(as_of_text)
except ValueError as error:
    st.sidebar.error(str(error))

(
    market_tab,
    forecast_tab,
    drivers_tab,
    intelligence_tab,
    agent_tab,
    evaluation_tab,
    quality_tab,
) = st.tabs(
    [
        "Market Dashboard",
        "Forecast",
        "Drivers",
        "Intelligence",
        "Ask AI",
        "Evaluation",
        "Data Quality",
    ]
)

with market_tab:
    st.subheader("Market snapshot")
    if as_of:
        try:
            snapshot = market.snapshot(as_of)
            weather = market.weather_signal(as_of, 14)
            columns = st.columns(4)
            columns[0].metric("Henry Hub", f"${snapshot['latest_price']:.2f}/MMBtu")
            columns[1].metric(
                "Storage",
                f"{snapshot['storage_bcf']:,.0f} Bcf"
                if snapshot["storage_bcf"] is not None
                else "Unavailable",
            )
            columns[2].metric(
                "Storage vs 5-year",
                f"{snapshot['storage_vs_5yr_avg_bcf']:+,.0f} Bcf"
                if snapshot["storage_vs_5yr_avg_bcf"] is not None
                else "Unavailable",
            )
            anomaly = weather["mean_temperature_anomaly"]
            columns[3].metric(
                "14-day temp anomaly", f"{anomaly:+.1f}°F" if anomaly is not None else "Unavailable"
            )
            prices = repository.prices_as_of(as_of)
            price_chart_data = prepare_price_chart_data(prices)
            st.altair_chart(
                build_henry_hub_price_chart(price_chart_data),
                use_container_width=True,
            )
            st.caption(
                f"Price observation: {snapshot['price_date']} · "
                f"14d HDD: {number(weather['hdd']):.1f} · 14d CDD: {number(weather['cdd']):.1f} · "
                f"Regions: {', '.join(cast(list[str], weather['regions'])) or 'none'}"
            )
            latest_forecasts = repository.list_forecasts(100)
            by_horizon: dict[int, list[ForecastRecord]] = {}
            for row in latest_forecasts:
                by_horizon.setdefault(row.horizon, []).append(row)
            if latest_forecasts:
                st.markdown("#### Latest forecasts")
                forecast_columns = st.columns(3)
                for column, horizon in zip(forecast_columns, (1, 5, 20), strict=True):
                    horizon_rows = by_horizon.get(horizon, [])
                    if horizon_rows:
                        latest = horizon_rows[0]
                        previous = horizon_rows[1] if len(horizon_rows) > 1 else None
                        delta = latest.p50 - previous.p50 if previous else None
                        column.metric(
                            f"{horizon}d P50",
                            f"${latest.p50:.3f}",
                            f"{delta:+.3f} vs prior" if delta is not None else None,
                        )
                        column.caption(f"P10–P90 ${latest.p10:.3f}–${latest.p90:.3f}")
            recent_reports = repository.reports_as_of(as_of)[:3]
            if recent_reports:
                st.markdown("#### Recent reports")
                for report in recent_reports:
                    st.markdown(
                        f"- [{report.title}]({report.url}) — {report.publisher}, "
                        f"{report.publication_timestamp.date()}"
                    )
        except ValueError as error:
            st.warning(str(error))

with forecast_tab:
    st.subheader("Numerical forecast")
    st.caption("Generated by the configured forecasting model; the LLM cannot set these values.")
    horizons = st.multiselect("Observed-business-day horizons", [1, 5, 20], [1, 5, 20])
    if st.button("Train and run XGBoost", type="primary", disabled=not as_of or not horizons):
        assert as_of is not None
        try:
            with st.spinner("Walk-forward backtesting, calibration, and final fit…"):
                st.session_state["forecast_rows"] = forecasts.run_forecast(as_of, horizons)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
    rows = st.session_state.get("forecast_rows", repository.list_forecasts(3))
    if rows:
        for column, row in zip(st.columns(len(rows[:3])), rows[:3], strict=False):
            column.metric(f"{row.horizon}-day P50", f"${row.p50:.3f}")
            column.caption(f"Calibrated P10–P90: ${row.p10:.3f}–${row.p90:.3f}")
        st.dataframe(
            [
                {
                    "forecast_id": row.forecast_id,
                    "as_of": row.as_of_timestamp,
                    "horizon": row.horizon,
                    "p10": row.p10,
                    "p50": row.p50,
                    "p90": row.p90,
                    "persistence": row.baseline_forecast,
                    "model": row.model_version,
                }
                for row in rows
            ],
            width="stretch",
            hide_index=True,
        )

with drivers_tab:
    st.subheader("TreeSHAP forecast drivers")
    rows = st.session_state.get("forecast_rows", repository.list_forecasts(1))
    if rows:
        selected = st.selectbox(
            "Forecast",
            rows,
            format_func=lambda row: (
                f"{row.horizon}d · {row.as_of_timestamp} · {row.forecast_id[:8]}"
            ),
        )
        st.dataframe(
            [driver.__dict__ for driver in selected.drivers], width="stretch", hide_index=True
        )
        metrics = selected.metrics
        metric_columns = st.columns(4)
        metric_columns[0].metric("Walk-forward MAE", f"{metrics.get('walk_forward_mae', 0):.3f}")
        metric_columns[1].metric(
            "Skill vs persistence", f"{metrics.get('walk_forward_skill_vs_naive', 0):+.1%}"
        )
        metric_columns[2].metric(
            "Interval coverage", f"{metrics.get('walk_forward_interval_coverage', 0):.1%}"
        )
        metric_columns[3].metric(
            "Directional accuracy", f"{metrics.get('walk_forward_directional_accuracy', 0):.1%}"
        )
        st.caption("Contributions explain the fitted model prediction, not market causality.")
    else:
        st.info("Run a forecast to see drivers.")

with intelligence_tab:
    st.subheader("Hybrid market-intelligence retrieval")
    st.caption("BM25 + TF-IDF + embeddings + reranking, with hard as-of and metadata filters.")
    query = st.text_input("Search reports", "Henry Hub storage weather outlook")
    filter_columns = st.columns(3)
    document_type = filter_columns[0].text_input("Document type")
    publisher = filter_columns[1].text_input("Publisher")
    top_k = filter_columns[2].number_input("Results", 1, 20, 5)
    if st.button("Search intelligence", disabled=not as_of):
        assert as_of is not None
        try:
            results = retriever.search(
                query,
                as_of,
                int(top_k),
                SearchFilters(
                    document_type=document_type.strip() or None,
                    publisher=publisher.strip() or None,
                ),
            )
            if not results:
                st.info("No matching reports were available at this point in time.")
            for result in results:
                with st.container(border=True):
                    st.markdown(f"#### {result.title}")
                    st.caption(
                        f"{result.publisher} · {result.publication_timestamp} · "
                        f"hybrid score {result.score:.3f} · vector {result.vector_score:.3f}"
                    )
                    st.write(result.excerpt)
                    st.markdown(f"[Source]({result.source_url}) · `{result.citation}`")
        except (ValueError, RuntimeError) as error:
            st.error(str(error))

with agent_tab:
    st.subheader("Ask the MCP-grounded analyst")
    st.caption(
        "The model can only access market facts through the same MCP tools exposed externally."
    )
    question = st.text_area(
        "Question", "What is the current 20-day outlook, its main drivers, and supporting evidence?"
    )
    if not os.getenv("OPENAI_API_KEY"):
        st.info("Set OPENAI_API_KEY to enable the live Responses API agent.")
    if st.button("Ask analyst", disabled=not as_of or not os.getenv("OPENAI_API_KEY")):
        assert as_of is not None
        try:
            from commodity_ai.agent import MCPResponsesAgent

            with st.spinner("Calling the analyst and its MCP tools…"):
                answer = MCPResponsesAgent().answer_sync(question, as_of.isoformat())
            st.markdown(answer.answer)
            with st.expander("MCP tool trace"):
                st.json(list(answer.tool_calls))
        except (ValueError, RuntimeError) as error:
            st.error(str(error))

with evaluation_tab:
    st.subheader("Out-of-sample evaluation")
    evaluations = repository.forecast_evaluations()
    if evaluations:
        count = len(evaluations)
        mae = sum(number(row["absolute_error"]) for row in evaluations) / count
        rmse = math.sqrt(sum(number(row["squared_error"]) for row in evaluations) / count)
        direction = sum(number(row["direction_correct"]) for row in evaluations) / count
        coverage = (
            sum(
                1.0
                if number(row["p10"]) <= number(row["actual_price"]) <= number(row["p90"])
                else 0.0
                for row in evaluations
            )
            / count
        )
        columns = st.columns(4)
        columns[0].metric("Realized MAE", f"{mae:.3f}")
        columns[1].metric("Realized RMSE", f"{rmse:.3f}")
        columns[2].metric("Direction accuracy", f"{direction:.1%}")
        columns[3].metric("Interval coverage", f"{coverage:.1%}")
        chronological = list(reversed(evaluations))
        st.line_chart(
            {
                "Forecast": [number(row["point_forecast"]) for row in chronological],
                "Actual": [number(row["actual_price"]) for row in chronological],
                "Persistence": [number(row["baseline_forecast"]) for row in chronological],
            },
            x_label="Evaluated forecasts",
            y_label="USD/MMBtu",
        )
        st.line_chart(
            {"Forecast error": [number(row["error"]) for row in chronological]},
            x_label="Evaluated forecasts",
            y_label="USD/MMBtu",
        )
        st.dataframe(evaluations, width="stretch", hide_index=True)
    else:
        st.info("No matured forecasts have been evaluated yet.")
    st.divider()
    st.subheader("RAG benchmark")
    evaluation_path = Path("data/rag_evaluation.jsonl")
    if evaluation_path.exists() and st.button("Run retrieval benchmark"):
        metrics = evaluate_retriever(retriever, load_evaluation_cases(evaluation_path))
        st.json(metrics)
    elif not evaluation_path.exists():
        st.caption("Add data/rag_evaluation.jsonl to run the benchmark.")

with quality_tab:
    st.subheader("Data quality and freshness")
    st.dataframe(quality, width="stretch", hide_index=True)
    missing = [row["dataset"] for row in quality if not row["row_count"]]
    if missing:
        st.warning(f"No rows yet: {', '.join(str(value) for value in missing)}")
    else:
        st.success("All platform datasets contain records.")
    st.caption(
        "Run `commodity-ai ingest-live` with EIA_API_KEY for live market inputs. "
        "Raw source snapshots are retained under data/raw/."
    )
