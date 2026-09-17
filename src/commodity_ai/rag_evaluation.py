from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .domain import parse_datetime
from .rag import MarketIntelligenceRetriever, SearchFilters


@dataclass(frozen=True)
class RAGEvaluationCase:
    question: str
    as_of: datetime
    relevant_report_ids: tuple[str, ...]
    filters: SearchFilters = field(default_factory=SearchFilters)


def load_evaluation_cases(path: str | Path) -> list[RAGEvaluationCase]:
    cases: list[RAGEvaluationCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            RAGEvaluationCase(
                row["question"],
                parse_datetime(row["as_of"]),
                tuple(row["relevant_report_ids"]),
                SearchFilters(**row.get("filters", {})),
            )
        )
    return cases


def evaluate_retriever(
    retriever: MarketIntelligenceRetriever,
    cases: list[RAGEvaluationCase],
    top_k: int = 5,
) -> dict[str, float]:
    if not cases:
        raise ValueError("RAG evaluation requires at least one case")
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    citation_presence: list[float] = []
    citation_correctness: list[float] = []
    for case in cases:
        results = retriever.search(case.question, case.as_of, top_k, case.filters)
        returned = [result.report_id for result in results]
        relevant = set(case.relevant_report_ids)
        matched = relevant.intersection(returned)
        recalls.append(len(matched) / len(relevant) if relevant else 1.0)
        ranks = [index + 1 for index, report_id in enumerate(returned) if report_id in relevant]
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
        precisions.append(len(matched) / len(returned) if returned else 0.0)
        citation_presence.append(
            sum(bool(result.citation and result.source_url) for result in results) / len(results)
            if results
            else 0.0
        )
        relevant_results = [result for result in results if result.report_id in relevant]
        citation_correctness.append(
            sum(bool(result.citation and result.source_url) for result in relevant_results)
            / len(relevant_results)
            if relevant_results
            else 0.0
        )
    count = len(cases)
    return {
        f"recall_at_{top_k}": sum(recalls) / count,
        "mrr": sum(reciprocal_ranks) / count,
        "context_precision": sum(precisions) / count,
        "citation_presence": sum(citation_presence) / count,
        "citation_correctness": sum(citation_correctness) / count,
        "cases": float(count),
    }
