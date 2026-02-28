"""
SolarGrid DZ — MQTT Listener
Subscribes to solar inverter and smart meter MQTT topics using Modbus-over-MQTT.
Translates inverter readings into structured telemetry and forwards to Kafka / InfluxDB.
"""

import json
import logging
import os
import ssl
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_TLS = os.getenv("MQTT_TLS", "false").lower() == "true"

# Topic structure: solargrid/{installation_id}/{device_type}/{measurement}
TOPIC_BASE = "solargrid"
TOPIC_INVERTER = f"{TOPIC_BASE}/+/inverter/+"
TOPIC_BATTERY = f"{TOPIC_BASE}/+/battery/+"
TOPIC_METER = f"{TOPIC_BASE}/+/meter/+"
TOPIC_WEATHER = f"{TOPIC_BASE}/+/weather/+"


def parse_inverter_payload(topic: str, payload: dict) -> Optional[dict]:
    """
    Parse an inverter MQTT message into a standard solar reading.
    Supports SMA, Fronius, and Huawei SUN2000 payload formats.
    """
    parts = topic.split("/")
    installation_id = parts[1] if len(parts) > 1 else "UNKNOWN"

    # Normalise different inverter vendor formats
    power_kw = (
        payload.get("P_AC", payload.get("pac", payload.get("active_power_w", 0))) / 1000
        if "active_power_w" in payload
        else payload.get("P_AC", payload.get("pac", 0))
    )
    dc_voltage = payload.get("V_DC", payload.get("vdc", payload.get("pv_voltage_v", 380)))
    dc_current = payload.get("I_DC", payload.get("idc", payload.get("pv_current_a", 0)))
    temperature = payload.get("T_dev", payload.get("temp", payload.get("device_temperature_c", 25)))

    return {
        "installation_id": installation_id,
        "power_kw": round(float(power_kw), 3),
        "dc_voltage_v": round(float(dc_voltage), 2),
        "dc_current_a": round(float(dc_current), 3),
        "temperature_c": round(float(temperature), 2),
        "timestamp": datetime.fromisoformat(payload.get("ts", datetime.utcnow().isoformat())),
        "raw": payload,
    }


def parse_battery_payload(topic: str, payload: dict) -> Optional[dict]:
    parts = topic.split("/")
    installation_id = parts[1] if len(parts) > 1 else "UNKNOWN"
    return {
        "installation_id": installation_id,
        "soc": float(payload.get("SOC", payload.get("soc", 50))) / 100,
        "power_kw": float(payload.get("P", payload.get("power_w", 0))) / 1000,
        "voltage_v": float(payload.get("V", payload.get("voltage_v", 48))),
        "temperature_c": float(payload.get("T", payload.get("temperature_c", 25))),
        "timestamp": datetime.fromisoformat(payload.get("ts", datetime.utcnow().isoformat())),
    }


def parse_meter_payload(topic: str, payload: dict) -> Optional[dict]:
    parts = topic.split("/")
    installation_id = parts[1] if len(parts) > 1 else "UNKNOWN"
    return {
        "installation_id": installation_id,
        "import_kw": float(payload.get("import_kw", payload.get("import_w", 0))) / (1 if "import_kw" in payload else 1000),
        "export_kw": float(payload.get("export_kw", payload.get("export_w", 0))) / (1 if "export_kw" in payload else 1000),
        "frequency_hz": float(payload.get("freq_hz", 50.0)),
        "timestamp": datetime.fromisoformat(payload.get("ts", datetime.utcnow().isoformat())),
    }


# ─────────────────────────────────────────────
# MQTT Client
# ─────────────────────────────────────────────

class SolarMQTTListener:
    """
    Listens to MQTT topics from solar inverters and smart meters.
    Callbacks receive parsed telemetry dicts.
    """

    def __init__(
        self,
        broker: str = MQTT_BROKER,
        port: int = MQTT_PORT,
        username: str = MQTT_USERNAME,
        password: str = MQTT_PASSWORD,
        tls: bool = MQTT_TLS,
        on_inverter=None,
        on_battery=None,
        on_meter=None,
    ):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.tls = tls
        self.on_inverter = on_inverter
        self.on_battery = on_battery
        self.on_meter = on_meter
        self._client = None
        self._connected = False

    def _build_client(self):
        try:
            import paho.mqtt.client as mqtt
            client = mqtt.Client(client_id="solargrid-backend", protocol=mqtt.MQTTv5)
            client.on_connect = self._on_connect
            client.on_message = self._on_message
            client.on_disconnect = self._on_disconnect

            if self.username:
                client.username_pw_set(self.username, self.password)
            if self.tls:
                client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

            return client
        except ImportError:
            logger.warning("paho-mqtt not installed. MQTT listener disabled.")
            return None

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self._connected = True
            logger.info(f"MQTT connected to {self.broker}:{self.port}")
            client.subscribe(TOPIC_INVERTER, qos=1)
            client.subscribe(TOPIC_BATTERY, qos=1)
            client.subscribe(TOPIC_METER, qos=1)
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        self._connected = False
        logger.warning(f"MQTT disconnected (rc={rc}). Will retry...")

    def _on_message(self, client, userdata, message):
        topic = message.topic
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON on topic {topic}")
            return

        try:
            if "/inverter/" in topic and self.on_inverter:
                parsed = parse_inverter_payload(topic, payload)
                if parsed:
                    self.on_inverter(parsed)
            elif "/battery/" in topic and self.on_battery:
                parsed = parse_battery_payload(topic, payload)
                if parsed:
                    self.on_battery(parsed)
            elif "/meter/" in topic and self.on_meter:
                parsed = parse_meter_payload(topic, payload)
                if parsed:
                    self.on_meter(parsed)
        except Exception as e:
            logger.error(f"Error processing MQTT message on {topic}: {e}", exc_info=True)

    def start(self, blocking: bool = False):
        """Connect to MQTT broker and start listening."""
        self._client = self._build_client()
        if self._client is None:
            logger.info("MQTT listener not started (paho-mqtt unavailable)")
            return

        try:
            self._client.connect(self.broker, self.port, keepalive=60)
            if blocking:
                self._client.loop_forever()
            else:
                self._client.loop_start()
                logger.info("MQTT listener started in background")
        except Exception as e:
            logger.warning(f"MQTT connection error: {e}")

    def stop(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            logger.info("MQTT listener stopped")

    def publish_dispatch(self, installation_id: str, action: str, kw: float):
        """Publish a dispatch command back to the inverter/BMS via MQTT."""
        if self._client and self._connected:
            topic = f"{TOPIC_BASE}/{installation_id}/dispatch/command"
            payload = json.dumps({"action": action, "kw": kw, "ts": datetime.utcnow().isoformat()})
            self._client.publish(topic, payload, qos=1)
            logger.info(f"Dispatch command published: {installation_id} → {action} {kw} kW")


# ─────────────────────────────────────────────
# Integration factory
# ─────────────────────────────────────────────

def start_mqtt_pipeline(influx_client, kafka_producer=None) -> SolarMQTTListener:
    """
    Start MQTT listener that writes telemetry to InfluxDB and optionally Kafka.
    """
    def on_inverter(msg):
        influx_client.write_solar_reading(**{k: v for k, v in msg.items() if k != "raw"})
        if kafka_producer:
            kafka_producer.publish_solar(msg["installation_id"], msg["power_kw"])

    def on_battery(msg):
        influx_client.write_battery_state(**msg)
        if kafka_producer:
            kafka_producer.publish_battery(msg["installation_id"], msg["soc"], msg["power_kw"])

    def on_meter(msg):
        influx_client.write_grid_flow(**msg)

    listener = SolarMQTTListener(
        on_inverter=on_inverter,
        on_battery=on_battery,
        on_meter=on_meter,
    )
    listener.start(blocking=False)
    return listener
