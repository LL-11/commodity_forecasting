from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..domain import PriceObservation, StorageObservation, utc_now

EIA_BASE_URL = "https://api.eia.gov/v2"
HENRY_HUB_DAILY_SERIES = "NG.RNGWHHD.D"
LOWER_48_STORAGE_WEEKLY_SERIES = "NG.NW2_EPG0_SWO_R48_BCF.W"


class EIAClient:
    """Minimal EIA API v2 client with explicit pagination and timeouts."""

    def __init__(self, api_key: str, timeout_seconds: float = 30.0) -> None:
        if not api_key.strip():
            raise ValueError("EIA API key is required")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def fetch_series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
        length: int = 5000,
    ) -> dict[str, Any]:
        if not 1 <= length <= 5000:
            raise ValueError("EIA page length must be between 1 and 5000")
        params: dict[str, str | int] = {
            "api_key": self.api_key,
            "length": length,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        url = f"{EIA_BASE_URL}/seriesid/{series_id}?{urlencode(params)}"
        request = Request(
            url, headers={"Accept": "application/json", "User-Agent": "commodity-ai/0.1"}
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"EIA request failed for series {series_id}: {error}") from error
        if "error" in payload:
            raise RuntimeError(f"EIA returned an error: {payload['error']}")
        if not isinstance(payload.get("response", {}).get("data"), list):
            raise TypeError("EIA response does not contain response.data")
        return payload

    def fetch_henry_hub_daily(
        self, start: str | None = None, end: str | None = None
    ) -> dict[str, Any]:
        return self.fetch_series(HENRY_HUB_DAILY_SERIES, start, end)

    def fetch_lower_48_storage_weekly(
        self, start: str | None = None, end: str | None = None
    ) -> dict[str, Any]:
        return self.fetch_series(LOWER_48_STORAGE_WEEKLY_SERIES, start, end)


def normalize_henry_hub(
    payload: dict[str, Any], ingestion_timestamp: datetime | None = None
) -> list[PriceObservation]:
    """Normalize EIA rows conservatively using first-seen time as publication time.

    The series response does not provide a row-level release timestamp. Assigning the
    ingestion time prevents a backtest from seeing a value before this system first
    observed it. A production pipeline should preserve every raw snapshot so revisions
    remain reconstructible.
    """
    ingested = ingestion_timestamp or utc_now()
    rows: list[PriceObservation] = []
    for item in payload.get("response", {}).get("data", []):
        value = item.get("value")
        period = item.get("period")
        if value in (None, "", "NA") or not period:
            continue
        rows.append(
            PriceObservation(
                observation_date=date.fromisoformat(period),
                price=float(value),
                publication_timestamp=ingested,
                ingestion_timestamp=ingested,
                source="EIA API v2",
            )
        )
    rows.sort(key=lambda row: row.observation_date)
    return rows


def normalize_storage(
    payload: dict[str, Any], ingestion_timestamp: datetime | None = None
) -> list[StorageObservation]:
    """Normalize Lower-48 weekly working-gas history and derive comparison fields."""
    ingested = ingestion_timestamp or utc_now()
    values: list[tuple[date, float]] = []
    for item in payload.get("response", {}).get("data", []):
        value = item.get("value")
        period = item.get("period")
        if value in (None, "", "NA") or not period:
            continue
        values.append((date.fromisoformat(period), float(value)))
    values.sort(key=lambda item: item[0])
    by_year_week = {
        (day.isocalendar().year, day.isocalendar().week): value for day, value in values
    }
    rows: list[StorageObservation] = []
    for index, (day, value) in enumerate(values):
        iso = day.isocalendar()
        historical = [
            by_year_week[(iso.year - offset, iso.week)]
            for offset in range(1, 6)
            if (iso.year - offset, iso.week) in by_year_week
        ]
        five_year_average = sum(historical) / len(historical) if historical else value
        last_year = by_year_week.get((iso.year - 1, iso.week))
        weekly_change = value - values[index - 1][1] if index else 0.0
        rows.append(
            StorageObservation(
                period_end=day,
                storage_bcf=value,
                weekly_change_bcf=weekly_change,
                five_year_average_bcf=five_year_average,
                last_year_bcf=last_year,
                publication_timestamp=ingested,
                ingestion_timestamp=ingested,
                source="EIA API v2",
            )
        )
    return rows
