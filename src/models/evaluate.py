import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error

# ---------------------------
# Logger setup
# ---------------------------
logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# ---------------------------
# Hybrid model evaluator
# ---------------------------
def evaluate_region(region_id, trainset, testset, model_dir):
    """Evaluate Prophet + XGBoost hybrid model for a single region"""
    logger.info(f"📊 Evaluating Region {region_id}...")

    # Filter region data
    test_region = testset[testset["region"] == region_id].copy()
    if test_region.empty:
        logger.warning(f"⚠️ Region {region_id} has no test data.")
        return None, None

    test_region["tpep_pickup_datetime"] = pd.to_datetime(test_region["tpep_pickup_datetime"])

    # Load models
    prophet_path = model_dir / f"prophet_region_{region_id}.joblib"
    xgb_path = model_dir / f"xgb_region_{region_id}.joblib"

    if not prophet_path.exists() or not xgb_path.exists():
        logger.warning(f"⚠️ Models for Region {region_id} not found. Skipping.")
        return None, None

    prophet_model = joblib.load(prophet_path)
    xgb_model = joblib.load(xgb_path)

    # Step 1️⃣ - Prophet hourly forecast
    test_hourly = (
        test_region
        .resample("1H", on="tpep_pickup_datetime")
        .sum()
        .reset_index()
        .rename(columns={"tpep_pickup_datetime": "ds"})
    )
    forecast = prophet_model.predict(test_hourly)[["ds", "yhat"]]
    logger.info(f"Forecast shape: {forecast.shape}, Test region shape: {test_region.shape}")


    # Map hourly predictions back to 15-min data
    test_region["hour"] = test_region["tpep_pickup_datetime"].dt.floor("H")
    test_region = pd.merge(test_region, forecast, left_on="hour", right_on="ds", how="left")
    test_region.drop(columns=["hour", "ds"], inplace=True)
    test_region.rename(columns={"yhat": "prophet_trend"}, inplace=True)

    # Step 2️⃣ - XGBoost features
    features = [
        "lag_1", "lag_2", "lag_3", "lag_4",
        "avg_pickups", "day_of_week",
        "rolling_mean_3", "rolling_mean_6",
        "rolling_std_3", "rolling_std_6",
        "prophet_trend"
    ]

    # Prepare data
    X_test = test_region[features].copy()
    y_true = test_region["total_pickups"].copy()
    X_test["prophet_trend"].fillna(method="bfill", inplace=True)

    # Step 3️⃣ - Predict using XGBoost
    y_pred = xgb_model.predict(X_test)

    # Step 4️⃣ - Calculate MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    logger.info(f"✅ Region {region_id}: MAPE = {mape:.4f}")

    return region_id, mape


# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent

    # Paths
    model_dir = root_path / "models/training"
    eval_dir = root_path / "models/evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_path = root_path / "data/processed/train.csv"
    test_path = root_path / "data/processed/test.csv"

    # Load datasets
    trainset = pd.read_csv(train_path, parse_dates=["tpep_pickup_datetime"])
    testset = pd.read_csv(test_path, parse_dates=["tpep_pickup_datetime"])

    logger.info(f"✅ Train shape: {trainset.shape}, Test shape: {testset.shape}")

    # Evaluate all regions
    regions = sorted(trainset["region"].unique())
    results = []

    for region_id in regions:
        result = evaluate_region(region_id, trainset, testset, model_dir)
        if result[0] is not None:
            results.append(result)

    # Save results
    region_results = pd.DataFrame(results, columns=["region", "mape"])
    region_results.sort_values("mape", inplace=True)
    avg_mape = region_results["mape"].mean()

    # Save and print summary
    results_path = eval_dir / "regionwise_mape.csv"
    region_results.to_csv(results_path, index=False)
    logger.info(f"📂 Results saved to: {results_path}")
    logger.info("--------------------------------------------------")
    logger.info("📊 Region-wise MAPE Results:")
    logger.info("\n" + str(region_results))
    logger.info("--------------------------------------------------")
    logger.info(f"🏁 Average MAPE across all regions: {avg_mape:.4f}")
