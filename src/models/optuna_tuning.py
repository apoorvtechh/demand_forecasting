import optuna
import pandas as pd
import json
import logging
import warnings
from pathlib import Path
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# ---------------------------
# Logger setup
# ---------------------------
logger = logging.getLogger("optuna_tuning")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# =====================================================
# ⚙️ Function: Generate Prophet Trend + Tune XGBoost
# =====================================================
def tune_xgboost_with_fixed_prophet(region_id, trainset, testset):
    """Use Prophet trend as feature; tune only XGBoost with Optuna."""
    logger.info(f"\n🚀 Tuning XGBoost (using fixed Prophet trend) for Region {region_id}...")

    # Filter region data
    train_region = trainset[trainset["region"] == region_id].copy()
    test_region = testset[testset["region"] == region_id].copy()

    if train_region.empty or test_region.empty:
        logger.warning(f"⚠️ Skipping region {region_id} — not enough data.")
        return region_id, None, None

    # Ensure datetime
    for df in [train_region, test_region]:
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    # ====================================================
    # 🧠 Step 1 — Prophet (fixed parameters, not tuned)
    # ====================================================
    train_hourly = (
        train_region.resample("1H", on="tpep_pickup_datetime")
        .sum()
        .reset_index()
        .rename(columns={"tpep_pickup_datetime": "ds", "total_pickups": "y"})
    )

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False

    )
    m.fit(train_hourly)

    # Predict hourly trend for all timestamps
    all_times = pd.concat([train_region, test_region])["tpep_pickup_datetime"]
    all_times_hourly = all_times.dt.floor("H").drop_duplicates().reset_index(drop=True)
    forecast = m.predict(pd.DataFrame({"ds": all_times_hourly}))[["ds", "yhat"]]

    # Merge Prophet trend to 15-min data
    all_data = pd.concat([train_region, test_region])
    all_data["hour"] = all_data["tpep_pickup_datetime"].dt.floor("H")
    hybrid_data = pd.merge(all_data, forecast, left_on="hour", right_on="ds", how="left")
    hybrid_data.drop(columns=["hour", "ds"], inplace=True)
    hybrid_data.rename(columns={"yhat": "prophet_trend"}, inplace=True)

    # ====================================================
    # ⚙️ Step 2 — Prepare Data for XGBoost
    # ====================================================
    features = [
        "lag_1", "lag_2", "lag_3", "lag_4",
        "avg_pickups", "day_of_week",
        "rolling_mean_3", "rolling_mean_6",
        "rolling_std_3", "rolling_std_6",
        "prophet_trend"
    ]
    target = "total_pickups"

    hybrid_train = hybrid_data[hybrid_data["tpep_pickup_datetime"] < "2016-03-01"]
    hybrid_test  = hybrid_data[hybrid_data["tpep_pickup_datetime"] >= "2016-03-01"]

    X_train, y_train = hybrid_train[features], hybrid_train[target]
    X_test, y_test   = hybrid_test[features], hybrid_test[target]

    X_train["prophet_trend"].fillna(method="bfill", inplace=True)
    X_test["prophet_trend"].fillna(method="bfill", inplace=True)

    # ====================================================
    # 🎯 Step 3 — Optuna Objective (Tune only XGBoost)
    # ====================================================
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, preds)
        return mape

    # ====================================================
    # ⚡ Step 4 — Run Optuna
    # ====================================================
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    best_params = study.best_params
    best_mape = study.best_value

    logger.info(f"✅ Region {region_id}: Best XGBoost MAPE = {best_mape:.4f}")
    return region_id, best_mape, best_params


# =====================================================
# 🔁 Main entry point
# =====================================================
if __name__ == "__main__":
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent

    # Input paths
    train_path = root_path / "data/processed/train.csv"
    test_path = root_path / "data/processed/test.csv"

    # Output
    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_path = model_dir / "optuna_hybrid_regionwise_results.csv"

    # Load data
    trainset = pd.read_csv(train_path, parse_dates=["tpep_pickup_datetime"])
    testset = pd.read_csv(test_path, parse_dates=["tpep_pickup_datetime"])
    logger.info(f"✅ Data loaded: Train shape {trainset.shape}, Test shape {testset.shape}")

    # Tune for each region
    regions = sorted(trainset["region"].unique())
    results = []

    for region_id in regions:
        region_id, mape, best_params = tune_xgboost_with_fixed_prophet(region_id, trainset, testset)
        if mape is not None:
            results.append({
                "region": region_id,
                "best_mape": mape,
                "best_params": json.dumps(best_params)
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.sort_values("best_mape", inplace=True)
    results_df.to_csv(results_path, index=False)

    avg_mape = results_df["best_mape"].mean()
    logger.info(f"📂 Results saved to: {results_path}")
    logger.info(f"🏁 Average XGBoost (Prophet as feature) MAPE: {avg_mape:.4f}")
