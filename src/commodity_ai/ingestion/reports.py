from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.request import Request, urlopen

from ..domain import Report, utc_now


@dataclass(frozen=True)
class ReportSource:
    title: str
    url: str
    document_type: str


DEFAULT_EIA_REPORTS = (
    ReportSource(
        "EIA Natural Gas Weekly Update",
        "https://www.eia.gov/naturalgas/weekly/",
        "weekly-update",
    ),
    ReportSource(
        "EIA Short-Term Energy Outlook: Natural Gas",
        "https://www.eia.gov/outlooks/steo/marketreview/natgas.php",
        "steo",
    ),
    ReportSource(
        "EIA Short-Term Energy Outlook Overview",
        "https://www.eia.gov/outlooks/steo/report/",
        "steo",
    ),
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "footer"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth and data.strip():
            self.parts.append(data.strip())


class EIAReportClient:
    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(self, source: ReportSource, observed_at: datetime | None = None) -> Report:
        ingested = observed_at or utc_now()
        request = Request(source.url, headers={"User-Agent": "commodity-ai/0.1"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            html = response.read().decode("utf-8", errors="replace")
        extractor = _TextExtractor()
        extractor.feed(html)
        text = re.sub(r"\s+", " ", " ".join(extractor.parts)).strip()
        if len(text) < 200:
            raise ValueError(f"report extraction returned too little text for {source.url}")
        publication = _publication_timestamp(text, ingested)
        digest = hashlib.sha256(f"{source.url}|{publication.date()}".encode()).hexdigest()[:16]
        return Report(
            report_id=f"eia-{digest}",
            title=source.title,
            publisher="U.S. Energy Information Administration",
            publication_timestamp=publication,
            url=source.url,
            document_type=source.document_type,
            raw_text=text,
        )


def _publication_timestamp(text: str, fallback: datetime) -> datetime:
    match = re.search(
        r"(?:Release Date|Released)\s*:?\s*([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
        text,
    )
    if not match:
        return fallback
    parsed = datetime.strptime(match.group(1), "%B %d, %Y").replace(tzinfo=UTC)
    # EIA pages generally omit the release time. Noon UTC is intentionally
    # conservative; the first-seen ingestion timestamp remains the upper bound.
    candidate = parsed.replace(hour=12)
    return min(candidate, fallback)
