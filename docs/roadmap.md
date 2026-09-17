# Delivery status

## Implemented MVP path

- Live EIA price/storage, Open-Meteo regional weather, and EIA report ingestion
- Direct XGBoost training/prediction, walk-forward backtesting, TreeSHAP, and baselines
- MLflow run tracking and intervals calibrated from out-of-sample residuals
- Persistent embeddings, hybrid vector/lexical retrieval, reranking, citations, and retrieval metrics
- OpenAI Responses API agent connected through the MCP Python client
- Streamlit market/forecast/driver/intelligence/agent/evaluation/quality views
- FastAPI, MCP streamable HTTP, MLflow, and dashboard containers

## Next acceptance gates

1. Expand the frozen RAG and agent evaluation corpus to 50–100 expert-reviewed questions.
2. Add answer-level citation correctness, faithfulness, relevance, and tool-selection scoring.
3. Add SARIMAX and publish horizon-by-horizon comparisons against both baselines.
4. Add production/LNG/pipeline fundamentals and regime-segmented forecast evaluation.
5. Add scheduled ingestion, authentication, secrets management, monitoring, and cloud deployment.

Deep learning, multi-agent orchestration, streaming infrastructure, and trading integrations remain out of scope until these gates pass.
