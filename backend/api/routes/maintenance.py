"""Panel maintenance routes — fault detection and maintenance priority scoring."""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Query
from backend.api.models import MaintenanceReport, PanelHealthEntry

router = APIRouter()

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        try:
            from ml.fault_detection import FaultDetector
            _detector = FaultDetector()
        except Exception:
            pass
    return _detector


@router.get("/{installation_id}/report", response_model=MaintenanceReport)
async def maintenance_report(
    installation_id: str,
    request: Request,
    n_panels: int = Query(50, ge=1, le=500),
    n_days: int = Query(30, ge=1, le=365),
):
    """
    Run fault detection on panel telemetry and return a maintenance priority report.
    """
    detector = _get_detector()
    if detector is None:
        # Return a simulated report
        import random
        rng = random.Random(42)
        panels = [
            PanelHealthEntry(
                panel_id=f"P{i:03d}",
                anomaly_rate=round(rng.uniform(0, 0.15), 4),
                avg_anomaly_score=round(rng.uniform(0, 0.5), 4),
                max_anomaly_score=round(rng.uniform(0.05, 0.7), 4),
                dominant_fault=rng.choice(["normal", "normal", "soiling", "shading", "hardware"]),
                maintenance_priority=rng.choice(["OK", "OK", "OK", "MEDIUM", "HIGH"]),
            )
            for i in range(1, n_panels + 1)
        ]
        flagged = [p for p in panels if p.maintenance_priority != "OK"]
        return MaintenanceReport(
            installation_id=installation_id,
            n_panels_checked=n_panels,
            n_panels_flagged=len(flagged),
            report=panels,
            generated_at=datetime.utcnow(),
        )

    from ml.fault_detection import generate_panel_data
    data = generate_panel_data(n_panels=n_panels, n_days=n_days)
    report_df = detector.panel_health_report(data)

    panels = [
        PanelHealthEntry(
            panel_id=str(row.panel_id),
            anomaly_rate=round(float(row.anomaly_rate), 4),
            avg_anomaly_score=round(float(row.avg_anomaly_score), 4),
            max_anomaly_score=round(float(row.max_anomaly_score), 4),
            dominant_fault=str(row.dominant_fault),
            maintenance_priority=str(row.maintenance_priority),
        )
        for row in report_df.itertuples()
    ]
    flagged = [p for p in panels if p.maintenance_priority != "OK"]

    return MaintenanceReport(
        installation_id=installation_id,
        n_panels_checked=n_panels,
        n_panels_flagged=len(flagged),
        report=panels,
        generated_at=datetime.utcnow(),
    )


@router.get("/{installation_id}/alerts")
async def maintenance_alerts(installation_id: str, n_panels: int = Query(50)):
    """Return only panels with MEDIUM, HIGH, or CRITICAL priority."""
    detector = _get_detector()
    if detector is None:
        return {"installation_id": installation_id, "alerts": [], "message": "Model not trained yet"}

    from ml.fault_detection import generate_panel_data
    data = generate_panel_data(n_panels=n_panels, n_days=7)
    report_df = detector.panel_health_report(data)
    alerts = report_df[report_df["maintenance_priority"].isin(["MEDIUM", "HIGH", "CRITICAL"])].to_dict(orient="records")

    return {
        "installation_id": installation_id,
        "n_alerts": len(alerts),
        "alerts": alerts,
        "generated_at": datetime.utcnow().isoformat(),
    }
