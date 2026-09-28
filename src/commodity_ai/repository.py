from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

from .domain import (
    Driver,
    ForecastRecord,
    PriceObservation,
    Report,
    ReportChunk,
    StorageObservation,
    WeatherObservation,
    parse_date,
    parse_datetime,
)

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS price_daily (
    observation_date TEXT NOT NULL,
    price REAL NOT NULL,
    unit TEXT NOT NULL,
    source TEXT NOT NULL,
    publication_timestamp TEXT NOT NULL,
    ingestion_timestamp TEXT NOT NULL,
    PRIMARY KEY (observation_date, publication_timestamp, source)
);
CREATE TABLE IF NOT EXISTS storage_weekly (
    period_end TEXT NOT NULL,
    publication_timestamp TEXT NOT NULL,
    ingestion_timestamp TEXT NOT NULL,
    storage_bcf REAL NOT NULL,
    weekly_change_bcf REAL NOT NULL,
    five_year_average_bcf REAL NOT NULL,
    last_year_bcf REAL,
    source TEXT NOT NULL,
    PRIMARY KEY (period_end, publication_timestamp, source)
);
CREATE TABLE IF NOT EXISTS weather_daily (
    observation_date TEXT NOT NULL,
    region TEXT NOT NULL,
    hdd REAL NOT NULL,
    cdd REAL NOT NULL,
    temperature_anomaly REAL NOT NULL,
    forecast_or_actual TEXT NOT NULL,
    publication_timestamp TEXT NOT NULL,
    ingestion_timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    avg_temperature REAL,
    PRIMARY KEY (observation_date, region, publication_timestamp, source)
);
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    publisher TEXT NOT NULL,
    publication_timestamp TEXT NOT NULL,
    url TEXT NOT NULL,
    document_type TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    commodity TEXT NOT NULL,
    region TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS report_chunks (
    chunk_id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    publisher TEXT NOT NULL,
    publication_timestamp TEXT NOT NULL,
    document_type TEXT NOT NULL,
    commodity TEXT NOT NULL,
    region TEXT NOT NULL,
    source_url TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT NOT NULL REFERENCES report_chunks(chunk_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    created_timestamp TEXT NOT NULL,
    PRIMARY KEY (chunk_id, provider)
);
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id TEXT PRIMARY KEY,
    as_of_timestamp TEXT NOT NULL,
    latest_data_timestamp TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    point_forecast REAL NOT NULL,
    p10 REAL NOT NULL,
    p50 REAL NOT NULL,
    p90 REAL NOT NULL,
    baseline_forecast REAL NOT NULL,
    created_timestamp TEXT NOT NULL,
    drivers_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_actuals (
    forecast_id TEXT PRIMARY KEY REFERENCES forecasts(forecast_id),
    actual_price REAL NOT NULL,
    error REAL NOT NULL,
    absolute_error REAL NOT NULL,
    squared_error REAL NOT NULL,
    direction_correct INTEGER NOT NULL,
    evaluated_timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_price_publication ON price_daily(publication_timestamp);
CREATE INDEX IF NOT EXISTS idx_storage_publication ON storage_weekly(publication_timestamp);
CREATE INDEX IF NOT EXISTS idx_weather_publication ON weather_daily(publication_timestamp);
CREATE INDEX IF NOT EXISTS idx_reports_publication ON reports(publication_timestamp);
CREATE INDEX IF NOT EXISTS idx_chunks_publication ON report_chunks(publication_timestamp);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON report_chunks(commodity, region, document_type, publisher);
"""


class MarketRepository:
    def __init__(self, database: str | Path = ":memory:") -> None:
        self.database = str(database)
        if self.database != ":memory:":
            Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        # Streamlit reruns and FastAPI's sync worker pool may use the shared
        # repository from different threads. SQLite is compiled in serialized
        # mode; disabling the Python affinity guard permits that adapter usage.
        self.connection = sqlite3.connect(self.database, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        weather_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(weather_daily)")
        }
        if "avg_temperature" not in weather_columns:
            self.connection.execute("ALTER TABLE weather_daily ADD COLUMN avg_temperature REAL")
            self.connection.commit()
        self._backfill_report_chunks()

    def close(self) -> None:
        self.connection.close()

    def add_prices(self, rows: Iterable[PriceObservation]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO price_daily VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row.observation_date.isoformat(),
                    row.price,
                    row.unit,
                    row.source,
                    row.publication_timestamp.isoformat(),
                    row.ingestion_timestamp.isoformat(),
                )
                for row in rows
            ],
        )
        self.connection.commit()

    def add_storage(self, rows: Iterable[StorageObservation]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO storage_weekly VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row.period_end.isoformat(),
                    row.publication_timestamp.isoformat(),
                    row.ingestion_timestamp.isoformat(),
                    row.storage_bcf,
                    row.weekly_change_bcf,
                    row.five_year_average_bcf,
                    row.last_year_bcf,
                    row.source,
                )
                for row in rows
            ],
        )
        self.connection.commit()

    def add_weather(self, rows: Iterable[WeatherObservation]) -> None:
        self.connection.executemany(
            """INSERT OR REPLACE INTO weather_daily
               (observation_date, region, hdd, cdd, temperature_anomaly,
                forecast_or_actual, publication_timestamp, ingestion_timestamp,
                source, avg_temperature)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    row.observation_date.isoformat(),
                    row.region,
                    row.hdd,
                    row.cdd,
                    row.temperature_anomaly,
                    row.forecast_or_actual,
                    row.publication_timestamp.isoformat(),
                    row.ingestion_timestamp.isoformat(),
                    row.source,
                    row.avg_temperature,
                )
                for row in rows
            ],
        )
        self.connection.commit()

    def add_reports(self, rows: Iterable[Report]) -> None:
        from .chunking import chunk_report

        reports = list(rows)
        self.connection.executemany(
            "INSERT OR REPLACE INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row.report_id,
                    row.title,
                    row.publisher,
                    row.publication_timestamp.isoformat(),
                    row.url,
                    row.document_type,
                    row.raw_text,
                    row.commodity,
                    row.region,
                )
                for row in reports
            ],
        )
        for report in reports:
            self.connection.execute(
                "DELETE FROM report_chunks WHERE report_id = ?", (report.report_id,)
            )
            self.connection.executemany(
                "INSERT INTO report_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.chunk_id,
                        chunk.report_id,
                        chunk.title,
                        chunk.publisher,
                        chunk.publication_timestamp.isoformat(),
                        chunk.document_type,
                        chunk.commodity,
                        chunk.region,
                        chunk.source_url,
                        chunk.chunk_number,
                        chunk.text,
                    )
                    for chunk in chunk_report(report)
                ],
            )
        self.connection.commit()

    def _backfill_report_chunks(self) -> None:
        missing = self.connection.execute(
            """SELECT r.* FROM reports r
               WHERE NOT EXISTS (
                   SELECT 1 FROM report_chunks c WHERE c.report_id = r.report_id
               )"""
        ).fetchall()
        if not missing:
            return
        self.add_reports(
            Report(
                row["report_id"],
                row["title"],
                row["publisher"],
                parse_datetime(row["publication_timestamp"]),
                row["url"],
                row["document_type"],
                row["raw_text"],
                row["commodity"],
                row["region"],
            )
            for row in missing
        )

    def prices_as_of(
        self, as_of: datetime, start: date | None = None, end: date | None = None
    ) -> list[PriceObservation]:
        clauses = ["publication_timestamp <= ?", "observation_date <= ?"]
        params: list[str] = [as_of.isoformat(), as_of.date().isoformat()]
        if start:
            clauses.append("observation_date >= ?")
            params.append(start.isoformat())
        if end:
            clauses.append("observation_date <= ?")
            params.append(end.isoformat())
        # Latest published revision for each observation date.
        query = f"""
            SELECT p.* FROM price_daily p
            JOIN (
                SELECT observation_date, MAX(publication_timestamp) pub
                FROM price_daily WHERE {" AND ".join(clauses)} GROUP BY observation_date
            ) latest ON p.observation_date = latest.observation_date
                    AND p.publication_timestamp = latest.pub
            ORDER BY p.observation_date
        """
        rows = self.connection.execute(query, params).fetchall()
        return [
            PriceObservation(
                parse_date(r["observation_date"]),
                r["price"],
                parse_datetime(r["publication_timestamp"]),
                parse_datetime(r["ingestion_timestamp"]),
                r["source"],
                r["unit"],
            )
            for r in rows
        ]

    def first_published_prices_as_of(self, as_of: datetime) -> list[PriceObservation]:
        """Return the first revision published for each observation date."""
        rows = self.connection.execute(
            """SELECT p.* FROM price_daily p
               WHERE p.publication_timestamp <= ? AND p.observation_date <= ?
                 AND p.rowid = (
                     SELECT first.rowid FROM price_daily first
                     WHERE first.observation_date = p.observation_date
                       AND first.publication_timestamp <= ?
                     ORDER BY first.publication_timestamp ASC, first.source ASC
                     LIMIT 1
                 )
               ORDER BY p.observation_date""",
            (as_of.isoformat(), as_of.date().isoformat(), as_of.isoformat()),
        ).fetchall()
        return [
            PriceObservation(
                parse_date(row["observation_date"]),
                row["price"],
                parse_datetime(row["publication_timestamp"]),
                parse_datetime(row["ingestion_timestamp"]),
                row["source"],
                row["unit"],
            )
            for row in rows
        ]

    def latest_price_publication_timestamp(self) -> datetime | None:
        row = self.connection.execute(
            "SELECT MAX(publication_timestamp) latest FROM price_daily"
        ).fetchone()
        return parse_datetime(row["latest"]) if row and row["latest"] else None

    def latest_storage_as_of(self, as_of: datetime) -> StorageObservation | None:
        row = self.connection.execute(
            """SELECT * FROM storage_weekly WHERE publication_timestamp <= ?
               ORDER BY period_end DESC, publication_timestamp DESC LIMIT 1""",
            (as_of.isoformat(),),
        ).fetchone()
        if not row:
            return None
        return StorageObservation(
            parse_date(row["period_end"]),
            row["storage_bcf"],
            row["weekly_change_bcf"],
            row["five_year_average_bcf"],
            row["last_year_bcf"],
            parse_datetime(row["publication_timestamp"]),
            parse_datetime(row["ingestion_timestamp"]),
            row["source"],
        )

    def storage_history_as_of(self, as_of: datetime) -> list[StorageObservation]:
        """Return the latest known revision of each storage period."""
        rows = self.connection.execute(
            """SELECT s.* FROM storage_weekly s
               JOIN (
                   SELECT period_end, MAX(publication_timestamp) pub
                   FROM storage_weekly
                   WHERE publication_timestamp <= ?
                   GROUP BY period_end
               ) latest ON s.period_end = latest.period_end
                       AND s.publication_timestamp = latest.pub
               ORDER BY s.period_end""",
            (as_of.isoformat(),),
        ).fetchall()
        return [
            StorageObservation(
                parse_date(row["period_end"]),
                row["storage_bcf"],
                row["weekly_change_bcf"],
                row["five_year_average_bcf"],
                row["last_year_bcf"],
                parse_datetime(row["publication_timestamp"]),
                parse_datetime(row["ingestion_timestamp"]),
                row["source"],
            )
            for row in rows
        ]

    def weather_as_of(self, as_of: datetime, through: date) -> list[WeatherObservation]:
        rows = self.connection.execute(
            """SELECT w.* FROM weather_daily w
               JOIN (SELECT observation_date, region, MAX(publication_timestamp) pub
                     FROM weather_daily
                     WHERE publication_timestamp <= ? AND observation_date <= ?
                     GROUP BY observation_date, region) latest
               ON w.observation_date=latest.observation_date AND w.region=latest.region
                  AND w.publication_timestamp=latest.pub
               ORDER BY w.observation_date, w.region""",
            (as_of.isoformat(), through.isoformat()),
        ).fetchall()
        return [
            WeatherObservation(
                parse_date(r["observation_date"]),
                r["region"],
                r["hdd"],
                r["cdd"],
                r["temperature_anomaly"],
                r["forecast_or_actual"],
                parse_datetime(r["publication_timestamp"]),
                parse_datetime(r["ingestion_timestamp"]),
                r["source"],
                r["avg_temperature"],
            )
            for r in rows
        ]

    def reports_as_of(self, as_of: datetime) -> list[Report]:
        rows = self.connection.execute(
            "SELECT * FROM reports WHERE publication_timestamp <= ? ORDER BY publication_timestamp DESC",
            (as_of.isoformat(),),
        ).fetchall()
        return [
            Report(
                r["report_id"],
                r["title"],
                r["publisher"],
                parse_datetime(r["publication_timestamp"]),
                r["url"],
                r["document_type"],
                r["raw_text"],
                r["commodity"],
                r["region"],
            )
            for r in rows
        ]

    def get_report(self, report_id: str) -> Report | None:
        row = self.connection.execute(
            "SELECT * FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        if not row:
            return None
        return Report(
            row["report_id"],
            row["title"],
            row["publisher"],
            parse_datetime(row["publication_timestamp"]),
            row["url"],
            row["document_type"],
            row["raw_text"],
            row["commodity"],
            row["region"],
        )

    def report_chunks_as_of(
        self,
        as_of: datetime,
        *,
        commodity: str | None = None,
        region: str | None = None,
        document_type: str | None = None,
        publisher: str | None = None,
    ) -> list[ReportChunk]:
        clauses = ["publication_timestamp <= ?"]
        params: list[str] = [as_of.isoformat()]
        for column, value in (
            ("commodity", commodity),
            ("region", region),
            ("document_type", document_type),
            ("publisher", publisher),
        ):
            if value:
                clauses.append(f"LOWER({column}) = LOWER(?)")
                params.append(value)
        rows = self.connection.execute(
            f"""SELECT * FROM report_chunks WHERE {" AND ".join(clauses)}
                ORDER BY publication_timestamp DESC, report_id, chunk_number""",
            params,
        ).fetchall()
        return [
            ReportChunk(
                row["chunk_id"],
                row["report_id"],
                row["title"],
                row["publisher"],
                parse_datetime(row["publication_timestamp"]),
                row["document_type"],
                row["commodity"],
                row["region"],
                row["source_url"],
                row["chunk_number"],
                row["text"],
            )
            for row in rows
        ]

    def save_chunk_embeddings(
        self,
        provider: str,
        rows: Iterable[tuple[str, list[float]]],
        created_timestamp: datetime,
    ) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO chunk_embeddings VALUES (?, ?, ?, ?, ?)",
            [
                (chunk_id, provider, len(vector), json.dumps(vector), created_timestamp.isoformat())
                for chunk_id, vector in rows
            ],
        )
        self.connection.commit()

    def chunk_embeddings(self, provider: str, chunk_ids: Iterable[str]) -> dict[str, list[float]]:
        identifiers = list(chunk_ids)
        if not identifiers:
            return {}
        placeholders = ",".join("?" for _ in identifiers)
        rows = self.connection.execute(
            f"""SELECT chunk_id, vector_json FROM chunk_embeddings
                WHERE provider = ? AND chunk_id IN ({placeholders})""",
            [provider, *identifiers],
        ).fetchall()
        return {row["chunk_id"]: json.loads(row["vector_json"]) for row in rows}

    def save_forecast(self, forecast: ForecastRecord) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                forecast.forecast_id,
                forecast.as_of_timestamp.isoformat(),
                forecast.latest_data_timestamp.isoformat(),
                forecast.model_version,
                forecast.horizon,
                forecast.point_forecast,
                forecast.p10,
                forecast.p50,
                forecast.p90,
                forecast.baseline_forecast,
                forecast.created_timestamp.isoformat(),
                json.dumps([d.__dict__ for d in forecast.drivers]),
                json.dumps(forecast.metrics),
            ),
        )
        self.connection.commit()

    def get_forecast(self, forecast_id: str) -> ForecastRecord | None:
        row = self.connection.execute(
            "SELECT * FROM forecasts WHERE forecast_id = ?", (forecast_id,)
        ).fetchone()
        if not row:
            return None
        return self._forecast_from_row(row)

    @staticmethod
    def _forecast_from_row(row: sqlite3.Row) -> ForecastRecord:
        return ForecastRecord(
            row["forecast_id"],
            parse_datetime(row["as_of_timestamp"]),
            parse_datetime(row["latest_data_timestamp"]),
            row["model_version"],
            row["horizon"],
            row["point_forecast"],
            row["p10"],
            row["p50"],
            row["p90"],
            row["baseline_forecast"],
            parse_datetime(row["created_timestamp"]),
            tuple(Driver(**d) for d in json.loads(row["drivers_json"])),
            json.loads(row["metrics_json"]),
        )

    def list_forecasts(self, limit: int = 200) -> list[ForecastRecord]:
        rows = self.connection.execute(
            """SELECT * FROM forecasts
               ORDER BY as_of_timestamp DESC, horizon ASC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._forecast_from_row(row) for row in rows]

    def forecast_evaluations(self, limit: int = 500) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """SELECT f.forecast_id, f.as_of_timestamp, f.model_version, f.horizon,
                      f.point_forecast, f.p10, f.p90, f.baseline_forecast,
                      a.actual_price, a.error, a.absolute_error, a.squared_error,
                      a.direction_correct, a.evaluated_timestamp
               FROM forecasts f JOIN forecast_actuals a ON f.forecast_id = a.forecast_id
               ORDER BY a.evaluated_timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def data_quality_summary(self) -> list[dict[str, object]]:
        definitions = (
            ("prices", "price_daily", "publication_timestamp"),
            ("storage", "storage_weekly", "publication_timestamp"),
            ("weather", "weather_daily", "publication_timestamp"),
            ("reports", "reports", "publication_timestamp"),
            ("report_chunks", "report_chunks", "publication_timestamp"),
            ("embeddings", "chunk_embeddings", "created_timestamp"),
            ("forecasts", "forecasts", "created_timestamp"),
            ("evaluations", "forecast_actuals", "evaluated_timestamp"),
        )
        return [
            {
                "dataset": label,
                **dict(
                    self.connection.execute(
                        f"SELECT COUNT(*) row_count, MAX({timestamp}) latest_timestamp FROM {table}"
                    ).fetchone()
                ),
            }
            for label, table, timestamp in definitions
        ]

    def save_forecast_actual(
        self,
        forecast_id: str,
        actual_price: float,
        error: float,
        direction_correct: bool,
        evaluated_timestamp: datetime,
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO forecast_actuals VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                forecast_id,
                actual_price,
                error,
                abs(error),
                error**2,
                int(direction_correct),
                evaluated_timestamp.isoformat(),
            ),
        )
        self.connection.commit()

    def get_forecast_actual(self, forecast_id: str) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT * FROM forecast_actuals WHERE forecast_id = ?", (forecast_id,)
        ).fetchone()
        return dict(row) if row else None
