"""
SolarGrid DZ — Solar Production Forecast Model
48-hour ahead solar generation forecasting using Gradient Boosting + NWP weather correction.
"""

import numpy as np
import pandas as pd
import joblib
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal and weather features for forecasting."""
    df = df.copy()

    # Temporal features
    df["hour"] = df.index.hour
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["weekday"] = df.index.weekday

    # Cyclical encoding for periodic features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Solar position (clear-sky irradiance approximation)
    solar_elevation = _solar_elevation(df.index)
    df["solar_elevation"] = solar_elevation
    df["clear_sky_irr"] = np.maximum(0, np.sin(np.radians(solar_elevation)) * 1000)

    # Cloud coverage impact
    if "cloud_cover" in df.columns:
        df["cloud_factor"] = 1 - (df["cloud_cover"] / 100) * 0.75
        df["effective_irr"] = df["clear_sky_irr"] * df["cloud_factor"]
    else:
        df["effective_irr"] = df["clear_sky_irr"]

    # Lagged production features
    if "production_kw" in df.columns:
        for lag in [1, 2, 3, 24, 48]:
            df[f"prod_lag_{lag}h"] = df["production_kw"].shift(lag)
        df["prod_rolling_3h"] = df["production_kw"].rolling(3).mean()
        df["prod_rolling_24h"] = df["production_kw"].rolling(24).mean()

    df.dropna(inplace=True)
    return df


def _solar_elevation(timestamps: pd.DatetimeIndex, lat: float = 36.75, lon: float = 3.05) -> np.ndarray:
    """Approximate solar elevation angle (degrees) for Algiers."""
    doy = timestamps.dayofyear
    hour_decimal = timestamps.hour + timestamps.minute / 60.0

    # Equation of time (minutes)
    B = np.radians((360 / 365) * (doy - 81))
    eqt = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    solar_noon = 12 - (lon / 15) - (eqt / 60)
    hour_angle = 15 * (hour_decimal - solar_noon)  # degrees

    # Declination
    declination = 23.45 * np.sin(np.radians((360 / 365) * (doy - 81)))

    # Elevation
    lat_rad = np.radians(lat)
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    elevation = np.degrees(
        np.arcsin(
            np.sin(lat_rad) * np.sin(dec_rad)
            + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
        )
    )
    return elevation


# ─────────────────────────────────────────────
# Synthetic Data Generation (for training)
# ─────────────────────────────────────────────

def generate_synthetic_data(
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    capacity_kw: float = 500.0,
    location: str = "algiers",
) -> pd.DataFrame:
    """Generate realistic synthetic solar production data for Algeria."""
    idx = pd.date_range(start=start, end=end, freq="1h")
    df = pd.DataFrame(index=idx)

    # Weather simulation
    rng = np.random.default_rng(seed=42)
    df["temperature_c"] = 25 + 10 * np.sin(2 * np.pi * idx.dayofyear / 365) + rng.normal(0, 2, len(idx))
    df["wind_speed_ms"] = np.abs(rng.normal(3, 2, len(idx)))
    df["humidity_pct"] = 50 + 20 * np.sin(2 * np.pi * idx.dayofyear / 365 + np.pi) + rng.normal(0, 5, len(idx))
    df["cloud_cover"] = np.clip(rng.beta(1.5, 3, len(idx)) * 100, 0, 100)

    # Solar irradiance with cloud correction
    elevation = _solar_elevation(idx)
    clear_sky = np.maximum(0, np.sin(np.radians(elevation)) * 1000)
    cloud_attenuation = 1 - (df["cloud_cover"] / 100) * 0.75
    ghi = clear_sky * cloud_attenuation + rng.normal(0, 10, len(idx))
    ghi = np.clip(ghi, 0, None)

    # Panel efficiency and degradation
    temp_coeff = -0.004  # -0.4%/°C above 25°C
    temp_factor = 1 + temp_coeff * (df["temperature_c"] - 25)
    efficiency = 0.20 * temp_factor  # 20% base efficiency

    df["ghi_wm2"] = ghi
    df["production_kw"] = np.clip(ghi * efficiency * (capacity_kw / 1000) * 5, 0, capacity_kw)
    df["production_kw"] += rng.normal(0, 2, len(idx))
    df["production_kw"] = np.clip(df["production_kw"], 0, capacity_kw)

    logger.info(f"Generated {len(df):,} hourly records from {start} to {end}")
    return df


# ─────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "solar_elevation", "clear_sky_irr", "effective_irr",
    "temperature_c", "wind_speed_ms", "humidity_pct", "cloud_cover",
    "prod_lag_1h", "prod_lag_2h", "prod_lag_3h",
    "prod_lag_24h", "prod_lag_48h",
    "prod_rolling_3h", "prod_rolling_24h",
]


def train(
    df: Optional[pd.DataFrame] = None,
    capacity_kw: float = 500.0,
    location: str = "algiers",
    model_name: str = "solar_forecast",
) -> dict:
    """Train the solar production forecasting model."""
    if df is None:
        logger.info("No data provided — generating synthetic training data...")
        df = generate_synthetic_data(capacity_kw=capacity_kw, location=location)

    logger.info("Engineering features...")
    df_feat = engineer_features(df)

    available_features = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available_features].values
    y = df_feat["production_kw"].values / capacity_kw  # normalise 0–1

    # Time-series split
    tscv = TimeSeriesSplit(n_splits=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    logger.info(f"Training on {len(X_train):,} samples, testing on {len(X_test):,} samples")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
            verbose=0,
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 0, 1)
    mae = mean_absolute_error(y_test, y_pred) * 100  # as % of capacity
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 100

    metrics = {
        "mae_pct_capacity": round(mae, 3),
        "rmse_pct_capacity": round(rmse, 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": available_features,
    }

    logger.info(f"MAE: {mae:.2f}% of peak capacity | RMSE: {rmse:.2f}%")

    # Save model
    model_path = MODEL_DIR / f"{model_name}.pkl"
    meta_path = MODEL_DIR / f"{model_name}_meta.pkl"
    joblib.dump(pipeline, model_path)
    joblib.dump({"capacity_kw": capacity_kw, "features": available_features, "metrics": metrics}, meta_path)
    logger.info(f"Model saved → {model_path}")

    return metrics


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

class SolarForecaster:
    """Load a trained model and generate 48-hour production forecasts."""

    def __init__(self, model_name: str = "solar_forecast"):
        model_path = MODEL_DIR / f"{model_name}.pkl"
        meta_path = MODEL_DIR / f"{model_name}_meta.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train() first."
            )

        self.pipeline = joblib.load(model_path)
        meta = joblib.load(meta_path)
        self.capacity_kw = meta["capacity_kw"]
        self.feature_cols = meta["features"]
        self.metrics = meta["metrics"]
        logger.info(f"Loaded {model_name} | capacity={self.capacity_kw} kW | MAE={self.metrics['mae_pct_capacity']}%")

    def forecast(
        self,
        weather_forecast: Optional[pd.DataFrame] = None,
        horizon_hours: int = 48,
    ) -> pd.DataFrame:
        """
        Generate a production forecast for the next `horizon_hours`.

        Args:
            weather_forecast: DataFrame indexed by hourly timestamps with weather columns.
                              If None, uses clear-sky simulation with average conditions.
            horizon_hours: Number of hours ahead to forecast (default 48).

        Returns:
            DataFrame with columns: [timestamp, forecast_kw, lower_kw, upper_kw]
        """
        now = pd.Timestamp.now().floor("h")
        future_idx = pd.date_range(now, periods=horizon_hours, freq="1h")

        if weather_forecast is None:
            # Build synthetic weather for the forecast window
            rng = np.random.default_rng(seed=int(now.timestamp()) % 1000)
            weather_forecast = pd.DataFrame(index=future_idx)
            weather_forecast["temperature_c"] = 25 + 8 * np.sin(2 * np.pi * future_idx.dayofyear / 365) + rng.normal(0, 1.5, horizon_hours)
            weather_forecast["wind_speed_ms"] = np.abs(rng.normal(3, 1.5, horizon_hours))
            weather_forecast["humidity_pct"] = 50 + rng.normal(0, 8, horizon_hours)
            weather_forecast["cloud_cover"] = np.clip(rng.beta(1.5, 3, horizon_hours) * 100, 0, 100)

        df_feat = engineer_features(weather_forecast)

        available = [c for c in self.feature_cols if c in df_feat.columns]
        X = df_feat[available].values

        y_norm = self.pipeline.predict(X)
        y_norm = np.clip(y_norm, 0, 1)
        forecast_kw = y_norm * self.capacity_kw

        # Uncertainty bounds (±15% P10/P90 heuristic)
        uncertainty = forecast_kw * 0.15
        result = pd.DataFrame({
            "timestamp": df_feat.index,
            "forecast_kw": np.round(forecast_kw, 2),
            "lower_kw": np.round(np.maximum(0, forecast_kw - uncertainty), 2),
            "upper_kw": np.round(forecast_kw + uncertainty, 2),
        })
        return result


# ─────────────────────────────────────────────
# Weather API Integration (Open-Meteo — free)
# ─────────────────────────────────────────────

def fetch_weather_forecast(lat: float = 36.75, lon: float = 3.05, hours: int = 48) -> Optional[pd.DataFrame]:
    """Fetch NWP weather forecast from Open-Meteo (no API key required)."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,cloudcover,windspeed_10m,relativehumidity_2m"
        f"&forecast_days=3&timezone=Africa%2FAlgiers"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        idx = pd.to_datetime(data["time"])
        df = pd.DataFrame({
            "temperature_c": data["temperature_2m"],
            "cloud_cover": data["cloudcover"],
            "wind_speed_ms": data["windspeed_10m"],
            "humidity_pct": data["relativehumidity_2m"],
        }, index=idx)
        return df.iloc[:hours]
    except Exception as exc:
        logger.warning(f"Weather API unavailable ({exc}), using synthetic forecast")
        return None


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SolarGrid DZ — Solar Forecast")
    parser.add_argument("--mode", choices=["train", "forecast"], default="train")
    parser.add_argument("--location", default="algiers")
    parser.add_argument("--capacity-kw", type=float, default=500.0)
    parser.add_argument("--horizon", type=int, default=48)
    args = parser.parse_args()

    if args.mode == "train":
        metrics = train(capacity_kw=args.capacity_kw, location=args.location)
        print("\nTraining complete:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    elif args.mode == "forecast":
        weather = fetch_weather_forecast(hours=args.horizon)
        forecaster = SolarForecaster()
        fc = forecaster.forecast(weather_forecast=weather, horizon_hours=args.horizon)
        print(f"\n48-hour forecast for {args.location}:")
        print(fc.to_string(index=False))
