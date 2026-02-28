"""SolarGrid DZ — Celery background task queue.
Handles model retraining, scheduled dispatches, and report generation.
"""

import logging
import os
from celery import Celery
from celery.schedules import crontab

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "solargrid",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Africa/Algiers",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,
    beat_schedule={
        # Retrain solar forecast model — every Sunday at 02:00
        "retrain-solar-forecast": {
            "task": "backend.tasks.retrain_solar_forecast",
            "schedule": crontab(hour=2, minute=0, day_of_week="sunday"),
        },
        # Retrain demand model — every Sunday at 03:00
        "retrain-demand-forecast": {
            "task": "backend.tasks.retrain_demand_forecast",
            "schedule": crontab(hour=3, minute=0, day_of_week="sunday"),
        },
        # Retrain fault detection — every Sunday at 04:00
        "retrain-fault-detection": {
            "task": "backend.tasks.retrain_fault_detection",
            "schedule": crontab(hour=4, minute=0, day_of_week="sunday"),
        },
        # Run dispatch cycle every hour
        "hourly-dispatch": {
            "task": "backend.tasks.run_dispatch_cycle",
            "schedule": crontab(minute=5),  # 5 minutes past every hour
        },
        # Generate carbon report — daily at midnight
        "daily-carbon-report": {
            "task": "backend.tasks.generate_daily_carbon_report",
            "schedule": crontab(hour=0, minute=10),
        },
        # Check maintenance alerts — every 6 hours
        "maintenance-check": {
            "task": "backend.tasks.run_maintenance_check",
            "schedule": crontab(hour="*/6", minute=30),
        },
    },
)


# ─────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def retrain_solar_forecast(self):
    """Retrain the solar production forecasting model with recent data."""
    try:
        from ml.solar_forecast import train, generate_synthetic_data
        logger.info("Starting solar forecast model retraining...")
        metrics = train()
        logger.info(f"Solar forecast retrained: MAE={metrics['mae_pct_capacity']}%")
        return metrics
    except Exception as exc:
        logger.error(f"Solar forecast retrain failed: {exc}")
        raise self.retry(exc=exc)


@app.task(bind=True, max_retries=3, default_retry_delay=300)
def retrain_demand_forecast(self, district_type: str = "residential"):
    """Retrain the demand forecasting model."""
    try:
        from ml.demand_forecast import train
        logger.info(f"Retraining demand forecast for {district_type}...")
        metrics = train(district_type=district_type)
        return metrics
    except Exception as exc:
        logger.error(f"Demand forecast retrain failed: {exc}")
        raise self.retry(exc=exc)


@app.task(bind=True, max_retries=3, default_retry_delay=300)
def retrain_fault_detection(self):
    """Retrain the panel fault detection model."""
    try:
        from ml.fault_detection import train, generate_panel_data
        logger.info("Retraining fault detection model...")
        data = generate_panel_data(n_panels=100)
        metrics = train(df=data)
        return metrics
    except Exception as exc:
        logger.error(f"Fault detection retrain failed: {exc}")
        raise self.retry(exc=exc)


@app.task
def run_dispatch_cycle():
    """Run hourly dispatch cycle for all installations."""
    try:
        from optimizer.dispatch_engine import create_default_engine
        engine = create_default_engine(n_districts=5, simulation_mode=True)
        results = engine.dispatch_all()
        logger.info(f"Dispatch cycle: {len(results['dispatches'])} installations processed")
        return {"dispatches": len(results["dispatches"]), "transfers": len(results["district_transfers"])}
    except Exception as exc:
        logger.error(f"Dispatch cycle failed: {exc}")
        raise


@app.task
def generate_daily_carbon_report():
    """Generate and store daily carbon offset report."""
    try:
        import numpy as np
        total_solar = np.random.uniform(8000, 12000)  # kWh (replace with InfluxDB query)
        co2_kg = total_solar * 0.83 * 0.65
        report = {
            "date": __import__("datetime").date.today().isoformat(),
            "total_solar_kwh": round(total_solar, 2),
            "co2_avoided_kg": round(co2_kg, 2),
            "self_consumption_pct": 83.0,
        }
        logger.info(f"Daily carbon report: {report}")
        return report
    except Exception as exc:
        logger.error(f"Carbon report failed: {exc}")
        raise


@app.task
def run_maintenance_check(n_panels: int = 50):
    """Run fault detection and alert on high-priority panels."""
    try:
        from ml.fault_detection import FaultDetector, generate_panel_data
        detector = FaultDetector()
        data = generate_panel_data(n_panels=n_panels, n_days=7)
        report = detector.panel_health_report(data)
        critical = report[report["maintenance_priority"].isin(["HIGH", "CRITICAL"])]
        if len(critical) > 0:
            logger.warning(f"MAINTENANCE ALERT: {len(critical)} panels need attention:\n{critical.to_string()}")
        return {"checked": n_panels, "alerts": len(critical)}
    except FileNotFoundError:
        logger.info("Maintenance check skipped: fault detection model not trained yet")
        return {"checked": 0, "alerts": 0}
    except Exception as exc:
        logger.error(f"Maintenance check failed: {exc}")
        raise
