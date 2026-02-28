"""
SolarGrid DZ — Panel Fault Detection
Anomaly detection using Isolation Forest + rule-based degradation scoring.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Fault Types
# ─────────────────────────────────────────────

FAULT_TYPES = {
    "shading": "Partial or full shading detected",
    "soiling": "Dust/soiling reducing output",
    "hardware": "Hardware/inverter fault",
    "degradation": "Long-term efficiency degradation",
    "mismatch": "String mismatch (diode failure)",
    "normal": "Normal operation",
}


# ─────────────────────────────────────────────
# Synthetic Data
# ─────────────────────────────────────────────

def generate_panel_data(
    n_panels: int = 50,
    n_days: int = 365,
    anomaly_rate: float = 0.07,
) -> pd.DataFrame:
    """
    Generate synthetic per-panel hourly telemetry data.
    Injects realistic fault patterns.
    """
    rng = np.random.default_rng(42)
    records = []
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="1h")

    for panel_id in range(1, n_panels + 1):
        # Baseline efficiency (some panels are naturally slightly worse)
        base_efficiency = rng.uniform(0.18, 0.21)
        degradation_rate = rng.uniform(0.001, 0.005) / 8760  # per hour

        # Introduce faults for some panels
        fault_panel = rng.random() < anomaly_rate * 3
        fault_type = rng.choice(["shading", "soiling", "hardware", "mismatch"]) if fault_panel else "normal"
        fault_start = rng.integers(0, len(idx) // 2) if fault_panel else None
        fault_duration = rng.integers(24, 720) if fault_panel else None

        for i, ts in enumerate(idx):
            hour = ts.hour
            doy = ts.dayofyear

            # Clear-sky irradiance (simplified)
            solar_angle = max(0, np.sin(np.pi * (hour - 6) / 12))
            clear_sky_ghi = solar_angle * 900
            cloud = abs(rng.normal(0.2, 0.15))
            ghi = clear_sky_ghi * (1 - cloud * 0.6)

            # Temperature (affects efficiency)
            panel_temp = 20 + 15 * np.sin(np.pi * doy / 365) + solar_angle * 25 + rng.normal(0, 2)

            # Degradation over time
            eff = base_efficiency * (1 - degradation_rate * i)

            # Fault effects
            fault_active = (
                fault_panel
                and fault_start is not None
                and fault_start <= i <= fault_start + fault_duration
            )
            fault_factor = 1.0
            if fault_active:
                if fault_type == "shading":
                    fault_factor = rng.uniform(0.2, 0.5)
                elif fault_type == "soiling":
                    fault_factor = rng.uniform(0.65, 0.85)
                elif fault_type == "hardware":
                    fault_factor = rng.uniform(0.0, 0.15)
                elif fault_type == "mismatch":
                    fault_factor = rng.uniform(0.4, 0.7)

            peak_power_w = 400  # 400 W panel
            expected_power_w = ghi * eff * (peak_power_w / 1000) * fault_factor
            actual_power_w = max(0, expected_power_w + rng.normal(0, 5))

            records.append({
                "timestamp": ts,
                "panel_id": f"P{panel_id:03d}",
                "ghi_wm2": round(ghi, 2),
                "panel_temp_c": round(panel_temp, 2),
                "power_w": round(actual_power_w, 2),
                "expected_power_w": round(expected_power_w / max(fault_factor, 1e-3), 2),
                "voltage_v": round(36 + rng.normal(0, 0.5), 2),
                "current_a": round(actual_power_w / max(36, 1), 3),
                "fault_type": fault_type if fault_active else "normal",
                "is_fault": 1 if fault_active else 0,
            })

    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    logger.info(f"Generated {len(df):,} panel records | fault rate: {df['is_fault'].mean():.2%}")
    return df


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "power_ratio",
    "power_smoothed",
    "power_deviation",
    "voltage_v",
    "current_a",
    "panel_temp_c",
    "ghi_wm2",
    "power_rolling_std",
    "efficiency_actual",
    "efficiency_deviation",
]


def engineer_panel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for anomaly detection."""
    df = df.copy()

    # Normalised metrics
    df["power_ratio"] = df["power_w"] / np.maximum(df["ghi_wm2"], 1)
    df["efficiency_actual"] = df["power_w"] / np.maximum(df["ghi_wm2"] * 0.4, 1)

    # Per-panel rolling statistics (group by panel)
    df["power_smoothed"] = df.groupby("panel_id")["power_w"].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    df["power_rolling_std"] = df.groupby("panel_id")["power_w"].transform(
        lambda x: x.rolling(24, min_periods=1).std().fillna(0)
    )

    # Deviation from expected
    df["power_deviation"] = (df["power_w"] - df["expected_power_w"]) / np.maximum(df["expected_power_w"], 1)
    df["efficiency_deviation"] = df.groupby("panel_id")["efficiency_actual"].transform(
        lambda x: (x - x.rolling(168, min_periods=24).mean()) / np.maximum(x.rolling(168, min_periods=24).std(), 1e-6)
    )

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(
    df: pd.DataFrame = None,
    model_name: str = "fault_detection",
    contamination: float = 0.07,
) -> dict:
    """Train Isolation Forest for panel anomaly detection."""
    if df is None:
        logger.info("Generating synthetic panel telemetry...")
        df = generate_panel_data()

    logger.info("Engineering panel features...")
    df_feat = engineer_panel_features(df)

    # Only daytime data (when panels should produce)
    df_feat = df_feat[df_feat["ghi_wm2"] > 50].copy()

    X = df_feat[FEATURE_COLS].values
    y_true = df_feat["is_fault"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_features=len(FEATURE_COLS),
            random_state=42,
            n_jobs=-1,
        )),
    ])

    logger.info(f"Training Isolation Forest on {len(X):,} samples...")
    pipeline.fit(X)

    # Evaluate (Isolation Forest outputs: -1=anomaly, 1=normal)
    raw_pred = pipeline.predict(X)
    y_pred = (raw_pred == -1).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_train": len(X),
        "contamination": contamination,
    }

    logger.info(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

    model_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump(pipeline, model_path)
    joblib.dump({"metrics": metrics, "features": FEATURE_COLS}, MODEL_DIR / f"{model_name}_meta.pkl")
    logger.info(f"Model saved → {model_path}")

    return metrics


# ─────────────────────────────────────────────
# Inference & Maintenance Scoring
# ─────────────────────────────────────────────

class FaultDetector:
    """Real-time panel fault detection and maintenance priority scoring."""

    def __init__(self, model_name: str = "fault_detection"):
        model_path = MODEL_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Run train() first.")

        self.pipeline = joblib.load(model_path)
        meta = joblib.load(MODEL_DIR / f"{model_name}_meta.pkl")
        self.feature_cols = meta["features"]
        self.metrics = meta["metrics"]

    def detect(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in panel telemetry.

        Returns:
            DataFrame with added columns: [is_anomaly, anomaly_score, maintenance_priority, fault_type_predicted]
        """
        df = engineer_panel_features(telemetry_df)
        df_day = df[df["ghi_wm2"] > 50].copy()

        if len(df_day) == 0:
            df["is_anomaly"] = 0
            df["anomaly_score"] = 0.0
            df["maintenance_priority"] = "OK"
            return df

        X = df_day[self.feature_cols].values
        raw_scores = self.pipeline.named_steps["iforest"].decision_function(
            self.pipeline.named_steps["scaler"].transform(X)
        )

        df_day["anomaly_score"] = -raw_scores  # higher = more anomalous
        df_day["is_anomaly"] = (self.pipeline.predict(X) == -1).astype(int)
        df_day["fault_type_predicted"] = df_day.apply(self._classify_fault, axis=1)
        df_day["maintenance_priority"] = df_day["anomaly_score"].apply(self._priority_label)

        return df_day

    @staticmethod
    def _classify_fault(row: pd.Series) -> str:
        """Rule-based fault classification on top of anomaly detection."""
        if row.get("is_anomaly", 0) == 0:
            return "normal"
        power_ratio = row.get("power_ratio", 0)
        power_dev = row.get("power_deviation", 0)
        eff_dev = row.get("efficiency_deviation", 0)

        if power_dev < -0.6:
            return "hardware"
        if -0.6 <= power_dev < -0.3:
            return "shading"
        if eff_dev < -2:
            return "degradation"
        return "soiling"

    @staticmethod
    def _priority_label(score: float) -> str:
        if score > 0.4:
            return "CRITICAL"
        if score > 0.2:
            return "HIGH"
        if score > 0.05:
            return "MEDIUM"
        return "OK"

    def panel_health_report(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate health report per panel for the field team."""
        detected = self.detect(telemetry_df)
        report = (
            detected.groupby("panel_id")
            .agg(
                anomaly_rate=("is_anomaly", "mean"),
                avg_anomaly_score=("anomaly_score", "mean"),
                max_anomaly_score=("anomaly_score", "max"),
                dominant_fault=("fault_type_predicted", lambda x: x.value_counts().index[0]),
                maintenance_priority=("maintenance_priority", lambda x: x.value_counts().index[0]),
            )
            .reset_index()
            .sort_values("avg_anomaly_score", ascending=False)
        )
        return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SolarGrid DZ — Fault Detection")
    parser.add_argument("--mode", choices=["train", "detect"], default="train")
    parser.add_argument("--n-panels", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "train":
        data = generate_panel_data(n_panels=args.n_panels)
        metrics = train(df=data)
        print("\nTraining complete:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    elif args.mode == "detect":
        data = generate_panel_data(n_panels=5, n_days=30)
        detector = FaultDetector()
        report = detector.panel_health_report(data)
        print("\nPanel Health Report:")
        print(report.to_string(index=False))
