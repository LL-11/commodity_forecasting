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
from commodity_ai.rolling_backtest import RollingBacktestService, TrackedRollingResult
from commodity_ai.services import ForecastService, MarketService
from commodity_ai.tracking import MLflowTracker

st.set_page_config(page_title="Henry Hub Intelligence", page_icon="🔥", layout="wide")
st.title("Henry Hub Commodity Intelligence")
st.caption(
    "XGBoost forecasts with explicit historical or point-in-time training, calibrated "
    "intervals, TreeSHAP drivers, hybrid retrieval, and an MCP-grounded analyst."
)


@st.cache_resource
def services() -> tuple[
    MarketRepository,
    MarketService,
    ForecastService,
    MarketIntelligenceRetriever,
    RollingBacktestService,
]:
    repository = MarketRepository(os.getenv("COMMODITY_AI_DB", "data/commodity_ai.db"))
    tracker = MLflowTracker()
    return (
        repository,
        MarketService(repository),
        ForecastService(repository, tracker=tracker),
        MarketIntelligenceRetriever(repository),
        RollingBacktestService(repository, tracker),
    )


repository, market, forecasts, retriever, rolling_backtests = services()


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
    model_evaluation_tab,
    quality_tab,
) = st.tabs(
    [
        "Market Dashboard",
        "Forecast",
        "Drivers",
        "Intelligence",
        "Ask AI",
        "Evaluation",
        "Model Evaluation",
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
    training_mode_label = st.selectbox(
        "Training mode",
        ("Historical (bulk EIA, non-vintage)", "Point-in-time"),
    )
    training_mode = (
        "historical"
        if training_mode_label == "Historical (bulk EIA, non-vintage)"
        else "point_in_time"
    )
    if training_mode == "historical":
        st.warning(
            "Historical mode uses the latest bulk EIA history. Performance metrics are "
            "non-vintage and may be revision-biased; unavailable historical weather "
            "vintages are omitted."
        )
    else:
        st.caption(
            "Point-in-time mode uses only data visible at each historical cutoff and "
            "requires enough accumulated snapshots."
        )
    horizons = st.multiselect("Observed-business-day horizons", [1, 5, 20], [1, 5, 20])
    if st.button("Train and run XGBoost", type="primary", disabled=not as_of or not horizons):
        assert as_of is not None
        try:
            with st.spinner("Walk-forward backtesting, calibration, and final fit…"):
                st.session_state["forecast_rows"] = forecasts.run_forecast(
                    as_of,
                    horizons,
                    training_mode=training_mode,
                )
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

with model_evaluation_tab:
    st.subheader("Rolling backtesting")
    st.caption("Evaluate the configured forecasting model over disjoint out-of-sample windows.")
    configuration_columns = st.columns(4)
    backtest_horizon = configuration_columns[0].selectbox(
        "Forecast horizon", (1, 5, 20), index=1, format_func=lambda value: f"{value} days"
    )
    train_window = configuration_columns[1].number_input(
        "Training window", min_value=40, value=40, step=10
    )
    test_window = configuration_columns[2].number_input(
        "Test window", min_value=1, value=10, step=1
    )
    step = configuration_columns[3].number_input("Window step", min_value=1, value=10, step=1)
    backtest_configuration = (
        int(backtest_horizon),
        int(train_window),
        int(test_window),
        int(step),
    )
    if st.button("Run rolling backtest", disabled=not as_of):
        backtest_run_name = (
            f"streamlit-{backtest_horizon}d-rolling-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
        try:
            with st.spinner("Running rolling out-of-sample evaluation…"):
                completed_backtest = rolling_backtests.run(
                    horizon=backtest_configuration[0],
                    train_window=backtest_configuration[1],
                    test_window=backtest_configuration[2],
                    step=backtest_configuration[3],
                    run_name=backtest_run_name,
                )
            st.session_state["rolling_backtest_result"] = completed_backtest
            st.session_state["rolling_backtest_run_name"] = backtest_run_name
            st.session_state["rolling_backtest_configuration"] = backtest_configuration
        except (ValueError, RuntimeError) as error:
            st.error(str(error))

    stored_backtest_result = cast(
        TrackedRollingResult | None,
        st.session_state.get("rolling_backtest_result"),
    )
    stored_configuration = cast(
        tuple[int, int, int, int] | None,
        st.session_state.get("rolling_backtest_configuration"),
    )
    backtest_result = (
        stored_backtest_result if stored_configuration == backtest_configuration else None
    )
    stored_run_name = cast(str | None, st.session_state.get("rolling_backtest_run_name"))
    trace_columns = st.columns(4)
    trace_columns[0].metric(
        "MLflow tracking", "Enabled" if rolling_backtests.tracker.enabled else "Disabled"
    )
    trace_columns[1].metric(
        "Experiment",
        rolling_backtests.tracker.experiment_name
        if rolling_backtests.tracker.enabled
        else "Not configured",
    )
    trace_columns[2].metric(
        "Run ID",
        backtest_result.parent_run_id
        if backtest_result and backtest_result.parent_run_id
        else "Not recorded",
    )
    trace_columns[3].metric(
        "Run name",
        stored_run_name
        if backtest_result and backtest_result.parent_run_id and stored_run_name
        else "Not recorded",
    )

    if backtest_result is None:
        st.info("Rolling backtest results are not available for this forecast configuration.")
    else:
        summary = backtest_result.evaluation.summary
        pooled_metrics = cast(dict[str, object], summary["pooled_metrics"])
        period_statistics = cast(dict[str, dict[str, object]], summary["period_statistics"])
        predictions = [
            prediction
            for period in backtest_result.evaluation.periods
            for prediction in period.predictions
        ]
        start_date = min(prediction.target_date for prediction in predictions)
        end_date = max(prediction.target_date for prediction in predictions)

        aggregate_columns = st.columns(5)
        aggregate_columns[0].metric("Rolling windows", int(number(summary["period_count"])))
        aggregate_columns[1].metric("Forecast horizon", f"{int(number(summary['horizon']))} days")
        aggregate_columns[2].metric("Aggregate MAE", f"{number(pooled_metrics['test_mae']):.3f}")
        aggregate_columns[3].metric("Aggregate RMSE", f"{number(pooled_metrics['test_rmse']):.3f}")
        aggregate_columns[4].metric("Aggregate MAPE", f"{number(pooled_metrics['test_mape']):.1%}")
        mean_columns = st.columns(3)
        mean_columns[0].metric(
            "Mean window MAE", f"{number(period_statistics['test_mae']['mean']):.3f}"
        )
        mean_columns[1].metric(
            "Mean window RMSE", f"{number(period_statistics['test_rmse']['mean']):.3f}"
        )
        mean_columns[2].metric(
            "Mean window MAPE", f"{number(period_statistics['test_mape']['mean']):.1%}"
        )
        st.caption(f"Out-of-sample target dates: {start_date} to {end_date}")

        window_rows = [
            {
                "window_index": period.window.window_index,
                "train_start": period.train_start,
                "train_end": period.train_end,
                "test_start": period.predictions[0].origin_date,
                "test_end": period.predictions[-1].origin_date,
                "horizon": backtest_result.evaluation.dataset.horizon,
                "test_mae": period.metrics["test_mae"],
                "test_rmse": period.metrics["test_rmse"],
                "test_mape": period.metrics["test_mape"],
            }
            for period in backtest_result.evaluation.periods
        ]
        st.markdown("#### Performance by rolling window")
        st.line_chart(
            window_rows,
            x="test_start",
            y="test_rmse",
            x_label="Backtest window",
            y_label="RMSE (USD/MMBtu)",
        )
        st.markdown("#### Actual vs predicted")
        st.line_chart(
            {
                "target_date": [prediction.target_date for prediction in predictions],
                "Actual": [prediction.actual for prediction in predictions],
                "Predicted": [prediction.predicted for prediction in predictions],
            },
            x="target_date",
            y=["Actual", "Predicted"],
            x_label="Out-of-sample target date",
            y_label="USD/MMBtu",
        )
        with st.expander("Rolling backtest windows"):
            st.dataframe(window_rows, width="stretch", hide_index=True)

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

st.divider()
st.caption(
    "Educational software only. Forecasts and backtests are not trading or investment advice. "
    "Historical-mode results are non-vintage and may be revision-biased."
)
