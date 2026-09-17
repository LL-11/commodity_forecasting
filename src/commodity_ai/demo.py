from __future__ import annotations

import math
import random
from datetime import UTC, date, datetime, time, timedelta

from .domain import PriceObservation, Report, StorageObservation, WeatherObservation
from .repository import MarketRepository


def business_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def seed_demo(repository: MarketRepository, count: int = 180) -> datetime:
    """Load deterministic synthetic data. This is never presented as live EIA data."""
    rng = random.Random(42)
    dates = business_days(date(2025, 1, 2), count)
    prices = []
    for index, day in enumerate(dates):
        seasonal = 0.45 * math.cos(2 * math.pi * index / 90)
        trend = 0.002 * index
        value = max(1.2, 2.8 + seasonal + trend + rng.gauss(0, 0.10))
        published = datetime.combine(day, time(22, 0), tzinfo=UTC)
        prices.append(
            PriceObservation(day, round(value, 4), published, published, "synthetic-demo")
        )
    repository.add_prices(prices)

    storage = []
    for index, day in enumerate(dates[::5]):
        level = 2200 + 24 * index + 260 * math.sin(2 * math.pi * index / 52)
        published = datetime.combine(day + timedelta(days=6), time(15, 30), tzinfo=UTC)
        storage.append(
            StorageObservation(
                day,
                level,
                35 + 15 * math.sin(index / 4),
                level + 80,
                level - 30,
                published,
                published,
                "synthetic-demo",
            )
        )
    repository.add_storage(storage)

    regions = ["Northeast", "Midwest", "South", "Texas", "West"]
    weather = []
    for index, day in enumerate(dates):
        publication = datetime.combine(day, time(5, 0), tzinfo=UTC)
        for offset in range(15):
            observed = day + timedelta(days=offset)
            for region_index, region in enumerate(regions):
                phase = 2 * math.pi * (index + offset) / 260
                anomaly = 2.0 * math.sin(phase + region_index / 3)
                temperature = 58 + 25 * math.sin(phase) + anomaly
                weather.append(
                    WeatherObservation(
                        observed,
                        region,
                        max(0, 65 - temperature),
                        max(0, temperature - 65),
                        anomaly,
                        "actual" if offset == 0 else "forecast",
                        publication,
                        publication,
                        "synthetic-demo",
                    )
                )
    repository.add_weather(weather)

    repository.add_reports(
        [
            Report(
                "demo-storage",
                "Synthetic weekly storage note",
                "Demo Publisher",
                prices[-30].publication_timestamp,
                "https://example.invalid/storage",
                "weekly-update",
                "Working gas storage remained below its synthetic five-year average while injections continued.",
            ),
            Report(
                "demo-weather",
                "Synthetic weather outlook",
                "Demo Publisher",
                prices[-20].publication_timestamp,
                "https://example.invalid/weather",
                "weather-outlook",
                "Forecast temperatures were warmer than normal, reducing modeled heating demand.",
            ),
        ]
    )
    return prices[-1].publication_timestamp
