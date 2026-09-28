from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import TYPE_CHECKING, TypedDict

from .domain import PriceObservation

if TYPE_CHECKING:
    import altair as alt


PRICE_CHART_OBSERVATIONS = 120


class PriceChartDatum(TypedDict):
    observation_date: str
    price: float


def prepare_price_chart_data(
    prices: Iterable[PriceObservation], limit: int = PRICE_CHART_OBSERVATIONS
) -> list[PriceChartDatum]:
    """Return the latest price observations in chronological chart order."""
    if limit < 1:
        raise ValueError("price chart observation limit must be positive")

    normalized: list[tuple[date, float]] = []
    for row in prices:
        observation_date = row.observation_date
        if isinstance(observation_date, datetime):
            observation_date = observation_date.date()
        if not isinstance(observation_date, date):
            raise TypeError("price chart observations must have a valid observation date")
        normalized.append((observation_date, row.price))

    normalized.sort(key=lambda row: row[0])
    return [
        {
            # A timezone-free midnight is reliably parsed as the original calendar
            # date by Vega-Lite. Passing a Python date through inline Altair data
            # serializes it as a nested object, which produces an empty x-domain.
            "observation_date": f"{observation_date.isoformat()}T00:00:00",
            "price": price,
        }
        for observation_date, price in normalized[-limit:]
    ]


def build_henry_hub_price_chart(data: list[PriceChartDatum]) -> alt.Chart:
    """Build the Market Snapshot price chart with a true temporal x-axis."""
    import altair as alt

    return (
        alt.Chart(alt.Data(values=data))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "observation_date:T",
                title="Date",
                sort="ascending",
                axis=alt.Axis(format="%Y-%m-%d", labelOverlap="greedy"),
            ),
            y=alt.Y("price:Q", title="USD/MMBtu"),
            tooltip=[
                alt.Tooltip(
                    "observation_date:T",
                    title="Observation date",
                    format="%Y-%m-%d",
                ),
                alt.Tooltip(
                    "price:Q",
                    title="Henry Hub ($/MMBtu)",
                    format="$.2f",
                ),
            ],
        )
    )
