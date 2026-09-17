from __future__ import annotations

import unittest
from datetime import UTC, datetime

from commodity_ai.ingestion.eia import normalize_henry_hub, normalize_storage
from commodity_ai.ingestion.weather import WeatherLocation, normalize_weather


class EIAIngestionTests(unittest.TestCase):
    def test_normalizer_skips_missing_values_and_tracks_first_seen_time(self) -> None:
        ingested = datetime(2025, 2, 1, 12, tzinfo=UTC)
        payload = {
            "response": {
                "data": [
                    {"period": "2025-01-02", "value": "3.21"},
                    {"period": "2025-01-03", "value": "NA"},
                ]
            }
        }
        rows = normalize_henry_hub(payload, ingested)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].price, 3.21)
        self.assertEqual(rows[0].publication_timestamp, ingested)

    def test_storage_normalizer_derives_changes_and_history_comparisons(self) -> None:
        ingested = datetime(2025, 2, 1, 12, tzinfo=UTC)
        data = []
        for year, value in ((2023, 2000), (2024, 2100), (2025, 2200)):
            data.append({"period": f"{year}-01-03", "value": str(value)})
        rows = normalize_storage({"response": {"data": data}}, ingested)
        self.assertEqual(rows[-1].storage_bcf, 2200)
        self.assertEqual(rows[-1].weekly_change_bcf, 100)
        self.assertEqual(rows[-1].last_year_bcf, 2100)
        self.assertEqual(rows[-1].five_year_average_bcf, 2050)

    def test_weather_normalizer_population_weights_degree_days(self) -> None:
        ingested = datetime(2025, 1, 1, 12, tzinfo=UTC)
        first = WeatherLocation("Test", "A", 1, 1, 0.75)
        second = WeatherLocation("Test", "B", 2, 2, 0.25)
        import commodity_ai.ingestion.weather as weather_module

        weather_module.REGIONAL_NORMALS_F["Test"] = (50,) * 12
        payloads = [
            (first, {"daily": {"time": ["2025-01-02"], "temperature_2m_mean": [40]}}),
            (second, {"daily": {"time": ["2025-01-02"], "temperature_2m_mean": [60]}}),
        ]
        rows = normalize_weather(payloads, ingested)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].avg_temperature, 45)
        self.assertEqual(rows[0].hdd, 20)
        self.assertEqual(rows[0].temperature_anomaly, -5)


if __name__ == "__main__":
    unittest.main()
