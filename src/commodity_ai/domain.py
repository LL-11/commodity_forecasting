from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.astimezone(UTC)


def parse_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


@dataclass(frozen=True)
class PriceObservation:
    observation_date: date
    price: float
    publication_timestamp: datetime
    ingestion_timestamp: datetime
    source: str = "EIA"
    unit: str = "USD/MMBtu"

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("price cannot be negative")
        parse_datetime(self.publication_timestamp)
        parse_datetime(self.ingestion_timestamp)


@dataclass(frozen=True)
class StorageObservation:
    period_end: date
    storage_bcf: float
    weekly_change_bcf: float
    five_year_average_bcf: float
    last_year_bcf: float | None
    publication_timestamp: datetime
    ingestion_timestamp: datetime
    source: str = "EIA"


@dataclass(frozen=True)
class WeatherObservation:
    observation_date: date
    region: str
    hdd: float
    cdd: float
    temperature_anomaly: float
    forecast_or_actual: str
    publication_timestamp: datetime
    ingestion_timestamp: datetime
    source: str = "demo-weather"
    avg_temperature: float | None = None

    def __post_init__(self) -> None:
        if self.forecast_or_actual not in {"forecast", "actual"}:
            raise ValueError("forecast_or_actual must be 'forecast' or 'actual'")


@dataclass(frozen=True)
class Report:
    report_id: str
    title: str
    publisher: str
    publication_timestamp: datetime
    url: str
    document_type: str
    raw_text: str
    commodity: str = "Henry Hub Natural Gas"
    region: str = "United States"


@dataclass(frozen=True)
class ReportChunk:
    chunk_id: str
    report_id: str
    title: str
    publisher: str
    publication_timestamp: datetime
    document_type: str
    commodity: str
    region: str
    source_url: str
    chunk_number: int
    text: str


@dataclass(frozen=True)
class Driver:
    feature: str
    effect: float
    direction: str


@dataclass(frozen=True)
class ForecastRecord:
    forecast_id: str
    as_of_timestamp: datetime
    latest_data_timestamp: datetime
    model_version: str
    horizon: int
    point_forecast: float
    p10: float
    p50: float
    p90: float
    baseline_forecast: float
    created_timestamp: datetime
    drivers: tuple[Driver, ...] = field(default_factory=tuple)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("as_of_timestamp", "latest_data_timestamp", "created_timestamp"):
            payload[key] = payload[key].isoformat()
        return payload
