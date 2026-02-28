"""
SolarGrid DZ — Kafka Consumer
Consumes real-time solar telemetry events from Kafka topics
and writes them to InfluxDB.
"""

import json
import logging
import os
import signal
import threading
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

TOPIC_SOLAR = "solar.production"
TOPIC_BATTERY = "battery.state"
TOPIC_GRID = "grid.flow"
TOPIC_WEATHER = "weather.update"
TOPIC_DISPATCH = "dispatch.command"
ALL_TOPICS = [TOPIC_SOLAR, TOPIC_BATTERY, TOPIC_GRID, TOPIC_WEATHER]

# ─────────────────────────────────────────────
# Message Schemas
# ─────────────────────────────────────────────

def parse_solar_message(payload: dict) -> dict:
    return {
        "installation_id": payload["installation_id"],
        "power_kw": float(payload["power_kw"]),
        "dc_voltage_v": float(payload.get("dc_voltage_v", 380)),
        "dc_current_a": float(payload.get("dc_current_a", 0)),
        "temperature_c": float(payload.get("temperature_c", 25)),
        "timestamp": datetime.fromisoformat(payload.get("ts", datetime.utcnow().isoformat())),
    }


def parse_battery_message(payload: dict) -> dict:
    return {
        "installation_id": payload["installation_id"],
        "soc": float(payload["soc"]),
        "power_kw": float(payload.get("power_kw", 0)),
        "voltage_v": float(payload.get("voltage_v", 48)),
        "temperature_c": float(payload.get("temperature_c", 25)),
        "timestamp": datetime.fromisoformat(payload.get("ts", datetime.utcnow().isoformat())),
    }


def parse_grid_message(payload: dict) -> dict:
    return {
        "installation_id": payload["installation_id"],
        "import_kw": float(payload.get("import_kw", 0)),
        "export_kw": float(payload.get("export_kw", 0)),
        "frequency_hz": float(payload.get("frequency_hz", 50.0)),
        "timestamp": datetime.fromisoformat(payload.get("ts", datetime.utcnow().isoformat())),
    }


# ─────────────────────────────────────────────
# Consumer (with kafka-python)
# ─────────────────────────────────────────────

class SolarKafkaConsumer:
    """
    Consumes solar telemetry events from Kafka and routes them to handlers.
    Falls back to MockConsumer if Kafka is unavailable.
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP,
        group_id: str = "solargrid-backend",
        on_solar: Optional[Callable] = None,
        on_battery: Optional[Callable] = None,
        on_grid: Optional[Callable] = None,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.on_solar = on_solar
        self.on_battery = on_battery
        self.on_grid = on_grid
        self._running = False
        self._consumer = None

    def _build_consumer(self):
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                *ALL_TOPICS,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=1000,
            )
            logger.info(f"Kafka consumer connected to {self.bootstrap_servers}")
            return consumer
        except Exception as e:
            logger.warning(f"Kafka unavailable ({e}). Using mock consumer.")
            return None

    def start(self, blocking: bool = True):
        """Start consuming messages. Set blocking=False to run in a background thread."""
        self._running = True
        self._consumer = self._build_consumer()

        if self._consumer is None:
            logger.info("Kafka not available — consumer idle")
            return

        def _run():
            for msg in self._consumer:
                if not self._running:
                    break
                self._dispatch(msg.topic, msg.value)

        if blocking:
            _run()
        else:
            t = threading.Thread(target=_run, daemon=True, name="kafka-consumer")
            t.start()
            return t

    def stop(self):
        self._running = False
        if self._consumer:
            self._consumer.close()

    def _dispatch(self, topic: str, payload: dict):
        try:
            if topic == TOPIC_SOLAR and self.on_solar:
                self.on_solar(parse_solar_message(payload))
            elif topic == TOPIC_BATTERY and self.on_battery:
                self.on_battery(parse_battery_message(payload))
            elif topic == TOPIC_GRID and self.on_grid:
                self.on_grid(parse_grid_message(payload))
        except Exception as e:
            logger.error(f"Error dispatching message on topic {topic}: {e}", exc_info=True)


# ─────────────────────────────────────────────
# Producer (for testing / simulation)
# ─────────────────────────────────────────────

class SolarKafkaProducer:
    """Publishes simulated solar telemetry events to Kafka."""

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP):
        self.bootstrap_servers = bootstrap_servers
        self._producer = None

    @property
    def producer(self):
        if self._producer is None:
            try:
                from kafka import KafkaProducer
                self._producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                )
            except Exception as e:
                logger.warning(f"Kafka producer unavailable: {e}")
        return self._producer

    def publish_solar(self, installation_id: str, power_kw: float, **kwargs):
        msg = {
            "installation_id": installation_id,
            "power_kw": power_kw,
            "ts": datetime.utcnow().isoformat(),
            **kwargs,
        }
        self._send(TOPIC_SOLAR, msg)

    def publish_battery(self, installation_id: str, soc: float, power_kw: float, **kwargs):
        msg = {
            "installation_id": installation_id,
            "soc": soc,
            "power_kw": power_kw,
            "ts": datetime.utcnow().isoformat(),
            **kwargs,
        }
        self._send(TOPIC_BATTERY, msg)

    def publish_dispatch_command(self, installation_id: str, action: str, kw: float):
        msg = {
            "installation_id": installation_id,
            "action": action,
            "kw": kw,
            "ts": datetime.utcnow().isoformat(),
        }
        self._send(TOPIC_DISPATCH, msg)

    def _send(self, topic: str, payload: dict):
        if self.producer:
            self.producer.send(topic, payload)
            logger.debug(f"Published to {topic}: {payload}")
        else:
            logger.debug(f"Mock publish to {topic}: {payload}")


# ─────────────────────────────────────────────
# Integration with InfluxDB
# ─────────────────────────────────────────────

def start_kafka_to_influx_pipeline(influx_client):
    """
    Start a Kafka consumer that writes telemetry directly to InfluxDB.
    Returns the consumer (stop it with consumer.stop()).
    """
    def on_solar(msg):
        influx_client.write_solar_reading(**msg)

    def on_battery(msg):
        influx_client.write_battery_state(**msg)

    def on_grid(msg):
        influx_client.write_grid_flow(**msg)

    consumer = SolarKafkaConsumer(
        on_solar=on_solar,
        on_battery=on_battery,
        on_grid=on_grid,
    )
    consumer.start(blocking=False)
    logger.info("Kafka → InfluxDB pipeline started")
    return consumer
