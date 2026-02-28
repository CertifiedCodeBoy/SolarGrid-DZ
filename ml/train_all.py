"""
SolarGrid DZ — Train All ML Models
Run this script once before starting the API to train all models.
"""

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_all")


def train_solar_forecast(location: str = "algiers", capacity_kw: float = 500.0):
    logger.info("=" * 60)
    logger.info("Training Solar Production Forecast Model")
    logger.info("=" * 60)
    from ml.solar_forecast import train
    t0 = time.time()
    metrics = train(capacity_kw=capacity_kw, location=location)
    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s | MAE: {metrics['mae_pct_capacity']}% of peak capacity")
    return metrics


def train_demand_forecast(district_type: str = "residential"):
    logger.info("=" * 60)
    logger.info(f"Training Demand Forecast Model ({district_type})")
    logger.info("=" * 60)
    from ml.demand_forecast import train
    t0 = time.time()
    metrics = train(district_type=district_type)
    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s | MAE: {metrics['mae_kw']} kW ({metrics['mae_pct_mean']}% of mean)")
    return metrics


def train_fault_detection(n_panels: int = 100):
    logger.info("=" * 60)
    logger.info("Training Panel Fault Detection Model")
    logger.info("=" * 60)
    from ml.fault_detection import train, generate_panel_data
    t0 = time.time()
    data = generate_panel_data(n_panels=n_panels)
    metrics = train(df=data)
    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s | Precision: {metrics['precision']} | Recall: {metrics['recall']}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all SolarGrid DZ ML models")
    parser.add_argument("--location", default="algiers")
    parser.add_argument("--capacity-kw", type=float, default=500.0)
    parser.add_argument("--n-panels", type=int, default=100)
    parser.add_argument("--district-type", default="residential",
                        choices=["residential", "commercial", "industrial", "hospital", "school"])
    parser.add_argument("--skip-solar", action="store_true")
    parser.add_argument("--skip-demand", action="store_true")
    parser.add_argument("--skip-fault", action="store_true")
    args = parser.parse_args()

    results = {}
    total_start = time.time()

    if not args.skip_solar:
        results["solar_forecast"] = train_solar_forecast(args.location, args.capacity_kw)

    if not args.skip_demand:
        results["demand_forecast"] = train_demand_forecast(args.district_type)

    if not args.skip_fault:
        results["fault_detection"] = train_fault_detection(args.n_panels)

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"All models trained in {total_elapsed:.1f}s")
    logger.info("Models saved to ml/saved_models/")
    logger.info("=" * 60)

    print("\nSummary:")
    for model, metrics in results.items():
        print(f"\n  [{model}]")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
