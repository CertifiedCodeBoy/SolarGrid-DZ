"""
SolarGrid DZ — InfluxDB Time-Series Client
Reads/writes real-time solar production and grid telemetry.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INFLUX_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-token")
INFLUX_ORG = os.getenv("INFLUXDB_ORG", "solargrid")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "solar_telemetry")


class InfluxDBManager:
    """Client wrapper for InfluxDB time-series operations."""

    def __init__(
        self,
        url: str = INFLUX_URL,
        token: str = INFLUX_TOKEN,
        org: str = INFLUX_ORG,
        bucket: str = INFLUX_BUCKET,
    ):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self._client: Optional[InfluxDBClient] = None

    @property
    def client(self) -> InfluxDBClient:
        if self._client is None:
            self._client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        return self._client

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    # ─── Write ──────────────────────────────

    def write_telemetry(self, records: list[dict]):
        """
        Write a batch of telemetry points to InfluxDB.

        Each record should have:
          - measurement: str (e.g. "solar_production", "battery_state", "grid_flow")
          - tags: dict
          - fields: dict
          - timestamp: datetime (optional, defaults to now)
        """
        write_api = self.client.write_api(write_options=SYNCHRONOUS)
        points = []
        for rec in records:
            p = (
                Point(rec["measurement"])
                .time(rec.get("timestamp", datetime.utcnow()))
            )
            for k, v in rec.get("tags", {}).items():
                p = p.tag(k, v)
            for k, v in rec.get("fields", {}).items():
                p = p.field(k, float(v) if isinstance(v, (int, float)) else v)
            points.append(p)
        write_api.write(bucket=self.bucket, org=self.org, record=points)
        logger.debug(f"Written {len(points)} points to InfluxDB")

    def write_solar_reading(
        self,
        installation_id: str,
        power_kw: float,
        dc_voltage_v: float,
        dc_current_a: float,
        temperature_c: float,
        timestamp: Optional[datetime] = None,
    ):
        """Write a single solar inverter reading."""
        self.write_telemetry([{
            "measurement": "solar_production",
            "tags": {"installation_id": installation_id},
            "fields": {
                "power_kw": power_kw,
                "dc_voltage_v": dc_voltage_v,
                "dc_current_a": dc_current_a,
                "temperature_c": temperature_c,
            },
            "timestamp": timestamp or datetime.utcnow(),
        }])

    def write_battery_state(
        self,
        installation_id: str,
        soc: float,
        power_kw: float,
        voltage_v: float,
        temperature_c: float,
        timestamp: Optional[datetime] = None,
    ):
        """Write battery state of charge and power."""
        self.write_telemetry([{
            "measurement": "battery_state",
            "tags": {"installation_id": installation_id},
            "fields": {
                "soc": soc,
                "power_kw": power_kw,
                "voltage_v": voltage_v,
                "temperature_c": temperature_c,
            },
            "timestamp": timestamp or datetime.utcnow(),
        }])

    def write_grid_flow(
        self,
        installation_id: str,
        import_kw: float,
        export_kw: float,
        frequency_hz: float,
        timestamp: Optional[datetime] = None,
    ):
        """Write grid import/export data."""
        self.write_telemetry([{
            "measurement": "grid_flow",
            "tags": {"installation_id": installation_id},
            "fields": {
                "import_kw": import_kw,
                "export_kw": export_kw,
                "frequency_hz": frequency_hz,
                "net_kw": export_kw - import_kw,
            },
            "timestamp": timestamp or datetime.utcnow(),
        }])

    # ─── Query ──────────────────────────────

    def query_production(
        self,
        installation_id: str,
        start: str = "-24h",
        stop: str = "now()",
        aggregate_window: str = "1h",
    ) -> pd.DataFrame:
        """Query solar production time series for an installation."""
        query = f"""
        from(bucket: "{self.bucket}")
          |> range(start: {start}, stop: {stop})
          |> filter(fn: (r) => r._measurement == "solar_production")
          |> filter(fn: (r) => r.installation_id == "{installation_id}")
          |> filter(fn: (r) => r._field == "power_kw")
          |> aggregateWindow(every: {aggregate_window}, fn: mean, createEmpty: false)
          |> yield(name: "mean")
        """
        return self._query_to_df(query)

    def query_battery_soc(
        self,
        installation_id: str,
        start: str = "-24h",
    ) -> pd.DataFrame:
        """Query battery SoC history."""
        query = f"""
        from(bucket: "{self.bucket}")
          |> range(start: {start})
          |> filter(fn: (r) => r._measurement == "battery_state")
          |> filter(fn: (r) => r.installation_id == "{installation_id}")
          |> filter(fn: (r) => r._field == "soc")
          |> yield(name: "soc")
        """
        return self._query_to_df(query)

    def query_energy_summary(
        self,
        district_id: Optional[str] = None,
        start: str = "-7d",
    ) -> dict:
        """Get aggregated energy totals for a district or the whole system."""
        tag_filter = f'|> filter(fn: (r) => r.district_id == "{district_id}")' if district_id else ""
        query = f"""
        from(bucket: "{self.bucket}")
          |> range(start: {start})
          |> filter(fn: (r) => r._measurement == "solar_production")
          {tag_filter}
          |> filter(fn: (r) => r._field == "power_kw")
          |> sum()
          |> yield(name: "total_kwh")
        """
        try:
            tables = self.client.query_api().query(query, org=self.org)
            total_kwh = sum(rec.get_value() for table in tables for rec in table.records)
            return {"total_solar_kwh": round(total_kwh, 2)}
        except Exception as e:
            logger.warning(f"InfluxDB query error: {e}")
            return {"total_solar_kwh": 0.0}

    def _query_to_df(self, query: str) -> pd.DataFrame:
        """Execute a Flux query and return a DataFrame."""
        try:
            df = self.client.query_api().query_data_frame(query, org=self.org)
            if isinstance(df, list):
                df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
            if not df.empty and "_time" in df.columns:
                df = df.rename(columns={"_time": "timestamp", "_value": "value"})
                df = df[["timestamp", "value"]].set_index("timestamp")
            return df
        except Exception as e:
            logger.warning(f"InfluxDB query error: {e}")
            return pd.DataFrame()


# ─────────────────────────────────────────────
# Simulated In-Memory Store (for development)
# ─────────────────────────────────────────────

class MockInfluxDB:
    """
    In-memory mock of InfluxDB for development without a running instance.
    Stores data in plain Python dicts.
    """

    def __init__(self):
        self._store: list[dict] = []
        logger.info("Using MockInfluxDB (in-memory). Set INFLUXDB_URL for real InfluxDB.")

    def write_telemetry(self, records: list[dict]):
        for rec in records:
            rec["_written_at"] = datetime.utcnow()
            self._store.append(rec)
        logger.debug(f"MockInfluxDB: wrote {len(records)} records (total={len(self._store)})")

    def write_solar_reading(self, installation_id, power_kw, **kwargs):
        self.write_telemetry([{
            "measurement": "solar_production",
            "tags": {"installation_id": installation_id},
            "fields": {"power_kw": power_kw, **kwargs},
        }])

    def write_battery_state(self, installation_id, soc, power_kw, **kwargs):
        self.write_telemetry([{
            "measurement": "battery_state",
            "tags": {"installation_id": installation_id},
            "fields": {"soc": soc, "power_kw": power_kw, **kwargs},
        }])

    def write_grid_flow(self, installation_id, import_kw, export_kw, **kwargs):
        self.write_telemetry([{
            "measurement": "grid_flow",
            "tags": {"installation_id": installation_id},
            "fields": {"import_kw": import_kw, "export_kw": export_kw, **kwargs},
        }])

    def query_production(self, installation_id, **kwargs) -> pd.DataFrame:
        rows = [
            {"timestamp": r["_written_at"], "value": r["fields"]["power_kw"]}
            for r in self._store
            if r["measurement"] == "solar_production"
            and r["tags"].get("installation_id") == installation_id
        ]
        return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()

    def query_energy_summary(self, **kwargs) -> dict:
        total = sum(
            r["fields"].get("power_kw", 0)
            for r in self._store
            if r["measurement"] == "solar_production"
        )
        return {"total_solar_kwh": round(total, 2)}


def get_influx_client() -> "InfluxDBManager | MockInfluxDB":
    """Return a real InfluxDB client or a mock depending on environment."""
    if os.getenv("INFLUXDB_URL"):
        return InfluxDBManager()
    return MockInfluxDB()
