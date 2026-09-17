from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..domain import WeatherObservation, utc_now

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@dataclass(frozen=True)
class WeatherLocation:
    region: str
    name: str
    latitude: float
    longitude: float
    weight: float


DEFAULT_LOCATIONS = (
    WeatherLocation("Northeast", "New York", 40.7128, -74.0060, 0.65),
    WeatherLocation("Northeast", "Boston", 42.3601, -71.0589, 0.35),
    WeatherLocation("Midwest", "Chicago", 41.8781, -87.6298, 0.60),
    WeatherLocation("Midwest", "Detroit", 42.3314, -83.0458, 0.40),
    WeatherLocation("South", "Atlanta", 33.7490, -84.3880, 0.55),
    WeatherLocation("South", "Charlotte", 35.2271, -80.8431, 0.45),
    WeatherLocation("Texas", "Houston", 29.7604, -95.3698, 0.55),
    WeatherLocation("Texas", "Dallas", 32.7767, -96.7970, 0.45),
    WeatherLocation("West", "Los Angeles", 34.0522, -118.2437, 0.45),
    WeatherLocation("West", "San Francisco", 37.7749, -122.4194, 0.25),
    WeatherLocation("West", "Seattle", 47.6062, -122.3321, 0.30),
)

# Approximate regional 1991-2020 monthly normal temperatures in Fahrenheit.
REGIONAL_NORMALS_F = {
    "Northeast": (31, 33, 42, 53, 63, 72, 77, 75, 68, 56, 46, 36),
    "Midwest": (25, 29, 40, 52, 63, 73, 77, 75, 67, 54, 40, 29),
    "South": (46, 49, 57, 65, 73, 80, 83, 82, 76, 66, 56, 49),
    "Texas": (48, 52, 61, 69, 77, 84, 87, 87, 80, 70, 59, 50),
    "West": (50, 52, 55, 59, 64, 69, 73, 73, 69, 62, 55, 50),
}


class OpenMeteoClient:
    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def _fetch(self, url: str, params: dict[str, str | int | float]) -> dict[str, Any]:
        request = Request(
            f"{url}?{urlencode(params)}",
            headers={"Accept": "application/json", "User-Agent": "commodity-ai/0.1"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload.get("daily", {}).get("time"), list):
            raise TypeError("Open-Meteo response does not contain daily observations")
        return payload

    def fetch_forecast(self, location: WeatherLocation, forecast_days: int = 14) -> dict[str, Any]:
        return self._fetch(
            OPEN_METEO_FORECAST_URL,
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "daily": "temperature_2m_mean",
                "temperature_unit": "fahrenheit",
                "timezone": "UTC",
                "forecast_days": forecast_days,
                "past_days": 7,
            },
        )

    def fetch_archive(
        self, location: WeatherLocation, start_date: date, end_date: date
    ) -> dict[str, Any]:
        return self._fetch(
            OPEN_METEO_ARCHIVE_URL,
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "daily": "temperature_2m_mean",
                "temperature_unit": "fahrenheit",
                "timezone": "UTC",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )


def normalize_weather(
    payloads: list[tuple[WeatherLocation, dict[str, Any]]],
    ingestion_timestamp: datetime | None = None,
) -> list[WeatherObservation]:
    ingested = ingestion_timestamp or utc_now()
    weighted: dict[tuple[str, date], list[tuple[float, float]]] = defaultdict(list)
    for location, payload in payloads:
        daily = payload["daily"]
        for day_text, temperature in zip(daily["time"], daily["temperature_2m_mean"], strict=True):
            if temperature is not None:
                weighted[(location.region, date.fromisoformat(day_text))].append(
                    (float(temperature), location.weight)
                )
    rows: list[WeatherObservation] = []
    for (region, day), observations in sorted(weighted.items()):
        total_weight = sum(weight for _, weight in observations)
        temperature = sum(value * weight for value, weight in observations) / total_weight
        normal = REGIONAL_NORMALS_F[region][day.month - 1]
        rows.append(
            WeatherObservation(
                observation_date=day,
                region=region,
                hdd=max(0.0, 65.0 - temperature),
                cdd=max(0.0, temperature - 65.0),
                temperature_anomaly=temperature - normal,
                forecast_or_actual="forecast" if day > ingested.date() else "actual",
                publication_timestamp=ingested,
                ingestion_timestamp=ingested,
                source="Open-Meteo",
                avg_temperature=temperature,
            )
        )
    return rows
