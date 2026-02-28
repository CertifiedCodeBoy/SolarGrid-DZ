"""
SolarGrid DZ — Demand Forecast Model
XGBoost-based hourly electricity demand forecasting.
Feeds the MPC optimizer with expected load profiles.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Optional

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Synthetic Demand Data
# ─────────────────────────────────────────────

DISTRICT_PROFILES = {
    "residential": {"peak_kw": 800, "morning_peak": 8, "evening_peak": 20, "base_load_frac": 0.25},
    "commercial": {"peak_kw": 1200, "morning_peak": 9, "evening_peak": 18, "base_load_frac": 0.15},
    "industrial": {"peak_kw": 1500, "morning_peak": 7, "evening_peak": 17, "base_load_frac": 0.40},
    "hospital": {"peak_kw": 600, "morning_peak": 8, "evening_peak": 14, "base_load_frac": 0.60},
    "school": {"peak_kw": 300, "morning_peak": 8, "evening_peak": 15, "base_load_frac": 0.05},
}


def generate_demand_data(
    district_type: str = "residential",
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    district_id: str = "DZ-ALG-01",
) -> pd.DataFrame:
    """Generate realistic hourly district electricity demand."""
    profile = DISTRICT_PROFILES.get(district_type, DISTRICT_PROFILES["residential"])
    idx = pd.date_range(start=start, end=end, freq="1h")
    rng = np.random.default_rng(42)

    # Base load
    demand = np.full(len(idx), profile["peak_kw"] * profile["base_load_frac"])

    # Morning peak (Gaussian)
    morning_peak = profile["peak_kw"] * 0.7 * np.exp(
        -0.5 * ((idx.hour - profile["morning_peak"]) / 2.5) ** 2
    )
    # Evening peak (Gaussian)
    evening_peak = profile["peak_kw"] * np.exp(
        -0.5 * ((idx.hour - profile["evening_peak"]) / 2.0) ** 2
    )

    demand += morning_peak + evening_peak

    # Seasonal variation (higher summer for cooling)
    seasonal = 0.2 * profile["peak_kw"] * np.sin(np.radians((idx.dayofyear - 80) * 360 / 365))
    demand += np.maximum(seasonal, 0)

    # Weekend reduction (except industrial/hospital)
    if district_type not in ("industrial", "hospital"):
        weekend_mask = idx.dayofweek >= 5
        demand[weekend_mask] *= 0.75

    # Ramadan effect (reduced commercial/industrial, slight residential shift)
    if district_type == "commercial":
        demand *= np.where((idx.month == 3) & (idx.day >= 10), 0.80, 1.0)

    # Temperature-dependent cooling load
    temp = 25 + 10 * np.sin(2 * np.pi * idx.dayofyear / 365) + rng.normal(0, 2, len(idx))
    cooling_load = np.maximum(0, (temp - 28) * 0.02 * profile["peak_kw"])
    demand += cooling_load

    # Noise
    demand += rng.normal(0, profile["peak_kw"] * 0.03, len(idx))
    demand = np.clip(demand, profile["peak_kw"] * profile["base_load_frac"] * 0.5, profile["peak_kw"] * 1.1)

    df = pd.DataFrame({
        "demand_kw": np.round(demand, 2),
        "temperature_c": np.round(temp, 2),
    }, index=idx)
    df.index.name = "timestamp"
    df["district_id"] = district_id
    df["district_type"] = district_type

    logger.info(f"Generated {len(df):,} hourly demand records for {district_id} ({district_type})")
    return df


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
    "month",
    "weekday",
    "is_weekend",
    "temperature_c",
    "temp_lag_1h", "temp_lag_24h",
    "demand_lag_1h", "demand_lag_24h", "demand_lag_168h",
    "demand_rolling_3h", "demand_rolling_24h",
    "demand_same_hour_last_week",
]


def engineer_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal and lag features for demand forecasting."""
    df = df.copy()

    df["hour"] = df.index.hour
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["weekday"] = df.index.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Lag features
    df["demand_lag_1h"] = df["demand_kw"].shift(1)
    df["demand_lag_24h"] = df["demand_kw"].shift(24)
    df["demand_lag_168h"] = df["demand_kw"].shift(168)  # 1 week
    df["demand_same_hour_last_week"] = df["demand_kw"].shift(168)

    df["temp_lag_1h"] = df["temperature_c"].shift(1)
    df["temp_lag_24h"] = df["temperature_c"].shift(24)

    # Rolling averages
    df["demand_rolling_3h"] = df["demand_kw"].rolling(3).mean()
    df["demand_rolling_24h"] = df["demand_kw"].rolling(24).mean()

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(
    df: Optional[pd.DataFrame] = None,
    district_type: str = "residential",
    model_name: str = "demand_forecast",
) -> dict:
    """Train XGBoost demand forecasting model."""
    if df is None:
        logger.info("Generating synthetic demand data...")
        df = generate_demand_data(district_type=district_type)

    logger.info("Engineering demand features...")
    df_feat = engineer_demand_features(df)

    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available].values
    y = df_feat["demand_kw"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    logger.info(f"Training XGBoost on {len(X_train):,} samples...")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    model.fit(
        X_train_sc, y_train,
        eval_set=[(X_test_sc, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test_sc)
    mae = mean_absolute_error(y_test, y_pred)
    mae_pct = mae / np.mean(y_test) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    peak = df_feat["demand_kw"].max()

    metrics = {
        "mae_kw": round(mae, 2),
        "mae_pct_peak": round(mae / peak * 100, 3),
        "mae_pct_mean": round(mae_pct, 3),
        "rmse_kw": round(rmse, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "peak_demand_kw": round(peak, 2),
    }

    logger.info(f"MAE: {mae:.1f} kW ({mae_pct:.2f}% of mean) | RMSE: {rmse:.1f} kW")

    model_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump({"model": model, "scaler": scaler, "features": available, "metrics": metrics}, model_path)
    logger.info(f"Model saved → {model_path}")
    return metrics


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

class DemandForecaster:
    """Load a trained XGBoost demand model and generate forecasts."""

    def __init__(self, model_name: str = "demand_forecast"):
        model_path = MODEL_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Run train() first.")

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["features"]
        self.metrics = bundle["metrics"]

    def forecast(self, recent_history: pd.DataFrame, horizon_hours: int = 48) -> pd.DataFrame:
        """
        Forecast demand for the next `horizon_hours` given recent historical data.

        Args:
            recent_history: DataFrame with at least 1 week of hourly demand data.
            horizon_hours: Forecast horizon.

        Returns:
            DataFrame with [timestamp, forecast_kw, lower_kw, upper_kw].
        """
        now = pd.Timestamp.now().floor("h")
        future_idx = pd.date_range(now + timedelta(hours=1), periods=horizon_hours, freq="1h")

        # Build a combined history + placeholder future
        future_df = pd.DataFrame(
            {"demand_kw": np.nan, "temperature_c": recent_history["temperature_c"].mean()},
            index=future_idx,
        )
        combined = pd.concat([recent_history, future_df]).sort_index()

        # Iterative multi-step forecast
        predictions = []
        for ts in future_idx:
            feat_df = engineer_demand_features(combined.loc[:ts].tail(180))
            if len(feat_df) == 0:
                predictions.append(recent_history["demand_kw"].mean())
                continue

            row = feat_df[self.feature_cols].iloc[[-1]].values
            pred = float(self.model.predict(self.scaler.transform(row))[0])
            pred = max(0, pred)
            predictions.append(pred)
            combined.loc[ts, "demand_kw"] = pred

        predictions = np.array(predictions)
        uncertainty = predictions * 0.08  # ±8%

        result = pd.DataFrame({
            "timestamp": future_idx,
            "forecast_kw": np.round(predictions, 2),
            "lower_kw": np.round(np.maximum(0, predictions - uncertainty), 2),
            "upper_kw": np.round(predictions + uncertainty, 2),
        })
        return result


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from datetime import timedelta

    parser = argparse.ArgumentParser(description="SolarGrid DZ — Demand Forecast")
    parser.add_argument("--mode", choices=["train", "forecast"], default="train")
    parser.add_argument("--district-type", default="residential",
                        choices=list(DISTRICT_PROFILES.keys()))
    args = parser.parse_args()

    if args.mode == "train":
        metrics = train(district_type=args.district_type)
        print("\nTraining complete:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    elif args.mode == "forecast":
        history = generate_demand_data(district_type=args.district_type, start="2025-01-01", end="2025-12-31")
        forecaster = DemandForecaster()
        result = forecaster.forecast(history, horizon_hours=48)
        print("\n48-hour demand forecast:")
        print(result.to_string(index=False))
