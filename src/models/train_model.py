import pandas as pd
import json
import joblib
import logging
from pathlib import Path
from prophet import Prophet
from xgboost import XGBRegressor

# ---------------------------
# Logger setup
# ---------------------------
logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ---------------------------
# Helper functions
# ---------------------------
def save_model(model, save_path):
    """Save model to disk."""
    joblib.dump(model, save_path)
    logger.info(f"💾 Model saved: {save_path}")

def load_optuna_params(optuna_path):
    """Load region-wise Optuna best parameters."""
    optuna_results = pd.read_csv(optuna_path)
    optuna_results['best_params'] = optuna_results['best_params'].apply(
        lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
    )
    logger.info("✅ Optuna region-wise parameters loaded successfully")
    return optuna_results

# ---------------------------
# Hybrid Prophet + XGBoost per region
# ---------------------------
def hybrid_prophet_xgb_region(region_id, trainset, testset, optuna_results, model_dir):
    """Train Prophet + XGBoost hybrid for one region."""
    logger.info(f"🚀 Training Hybrid Prophet + XGBoost (Optuna Params) for Region {region_id}")

    # Filter data
    train_region = trainset[trainset["region"] == region_id].copy()
    test_region = testset[testset["region"] == region_id].copy()
    if train_region.empty or test_region.empty:
        logger.warning(f"⚠️ Skipping region {region_id} — no data.")
        return None

    # Ensure datetime
    for df in [train_region, test_region]:
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    # ============================
    # 1️⃣ Train Prophet (hourly trend)
    # ============================
    train_hourly = (
        train_region.resample("1H", on="tpep_pickup_datetime")
        .sum()
        .reset_index()
    )
    prophet_train = train_hourly.rename(columns={"tpep_pickup_datetime": "ds", "total_pickups": "y"})

    m = Prophet(daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=False)       
    m.add_seasonality(name='daily_cycle', period=24, fourier_order=10)
    m.fit(prophet_train)

    # Predict hourly trend for all timestamps
    all_times = pd.concat([train_region, test_region])["tpep_pickup_datetime"]
    all_times_hourly = all_times.dt.floor("H").drop_duplicates().reset_index(drop=True)
    future = pd.DataFrame({"ds": all_times_hourly})
    forecast = m.predict(future)[["ds", "yhat"]]

    # Merge Prophet’s hourly yhat → 15-min data
    all_data = pd.concat([train_region, test_region])
    all_data["hour"] = all_data["tpep_pickup_datetime"].dt.floor("H")
    hybrid_data = pd.merge(all_data, forecast, left_on="hour", right_on="ds", how="left")
    hybrid_data.drop(columns=["hour", "ds"], inplace=True)
    hybrid_data.rename(columns={"yhat": "prophet_trend"}, inplace=True)

    # ============================
    # 2️⃣ Train XGBoost with Optuna params
    # ============================
    features = [
        "lag_1", "lag_2", "lag_3", "lag_4",
        "avg_pickups", "day_of_week",
        "rolling_mean_3", "rolling_mean_6",
        "rolling_std_3", "rolling_std_6",
        "prophet_trend"
    ]
    target = "total_pickups"

    hybrid_train = hybrid_data[hybrid_data["tpep_pickup_datetime"] < "2016-03-01"]
    X_train, y_train = hybrid_train[features], hybrid_train[target]

    X_train["prophet_trend"].fillna(method="bfill", inplace=True)

    # Get region’s tuned params
    best_params_row = optuna_results.loc[optuna_results["region"] == region_id]
    if best_params_row.empty:
        logger.warning(f"⚠️ No Optuna params found for region {region_id}, using defaults.")
        best_params = dict(n_estimators=300, learning_rate=0.05, max_depth=6)
    else:
        best_params = best_params_row["best_params"].values[0]

    model = XGBRegressor(**best_params,random_state=42)
    model.fit(X_train, y_train)

    # ============================
    # 💾 Save Models
    # ============================
    train_dir = model_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    prophet_path = train_dir / f"prophet_region_{region_id}.joblib"
    xgb_path = train_dir / f"xgb_region_{region_id}.joblib"

    save_model(m, prophet_path)
    save_model(model, xgb_path)

    logger.info(f"✅ Region {region_id}: Models saved successfully")

    return region_id


# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Data paths
    train_path = root_path / "data/processed/train.csv"
    test_path = root_path / "data/processed/test.csv"
    optuna_path = root_path / "models/optuna_hybrid_regionwise_results.csv"

    # Load data
    trainset = pd.read_csv(train_path, parse_dates=["tpep_pickup_datetime"])
    testset = pd.read_csv(test_path, parse_dates=["tpep_pickup_datetime"])
    logger.info(f"✅ Train shape: {trainset.shape}, Test shape: {testset.shape}")

    # Load Optuna parameters
    optuna_results = load_optuna_params(optuna_path)

    # Region loop
    regions = sorted(trainset["region"].unique())
    for region_id in regions:
        hybrid_prophet_xgb_region(region_id, trainset, testset, optuna_results, model_dir)

    logger.info("🏁 Training complete — all regional models saved successfully.")
