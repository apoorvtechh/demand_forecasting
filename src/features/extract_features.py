import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from yaml import safe_load
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_percentage_error

# ---------------- Logger ----------------
logger = logging.getLogger("extract_features")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ---------------- Utils ----------------
def read_params(params_path="params.yaml"):
    with open(params_path, "r") as file:
        params = safe_load(file)
    return params

def save_model(model, save_path):
    joblib.dump(model, save_path)
    logger.info(f"Model saved successfully at: {save_path}")

# ---------------- EWMA Optimization ----------------
def get_best_alpha_per_region(df, alphas):
    """Compute best alpha for each region using MAPE minimization."""
    results = []

    for region_id, region_data in df.groupby('region'):
        region_data = region_data.sort_index()
        alpha_scores = {}

        for alpha in alphas:
            y_pred = region_data['total_pickups'].ewm(alpha=alpha, adjust=False).mean().iloc[1:]
            y_true = region_data['total_pickups'].iloc[1:]
            if len(y_true) == 0:
                continue

            mape = mean_absolute_percentage_error(y_true, y_pred)
            alpha_scores[alpha] = mape

        if alpha_scores:
            best_alpha = min(alpha_scores, key=alpha_scores.get)
            best_mape = alpha_scores[best_alpha]
            results.append((region_id, best_alpha, best_mape))

    result_df = pd.DataFrame(results, columns=['region', 'best_alpha', 'best_mape'])
    return result_df

# ---------------- Main Pipeline ----------------
if __name__ == "__main__":
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent

    # 1️⃣ Load cleaned data
    data_path = root_path / "data/interim/df_cleaned.parquet"
    logger.info(f"📥 Reading data from: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Data loaded: {df.shape}")

    # 2️⃣ Initialize scaler and clusterer
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[["pickup_longitude", "pickup_latitude"]])
    logger.info("Scaler fitted successfully")

    params = read_params()["extract_features"]
    kmeans_params = params["mini_batch_kmeans"]

    mini_batch = MiniBatchKMeans(**kmeans_params)
    mini_batch.fit(scaled_features)
    logger.info("MiniBatchKMeans trained successfully")

    # 3️⃣ Save models
    save_model(scaler, root_path / "models/scaler.joblib")
    save_model(mini_batch, root_path / "models/mb_kmeans.joblib")

    # 4️⃣ Assign regions
    df["region"] = mini_batch.predict(scaled_features)
    df = df.drop(columns=["pickup_longitude", "pickup_latitude"])
    df.set_index("tpep_pickup_datetime", inplace=True)

    # 5️⃣ Resample to 15-min intervals
    region_grp = df.groupby("region")
    resampled = region_grp["region"].resample("15min").count()
    resampled.name = "total_pickups"
    resampled = resampled.reset_index(level=0)
    resampled["total_pickups"].replace(0, 10, inplace=True)  # Avoid zeros
    logger.info("Resampled 15-min total pickups successfully")

    # 6️⃣ Region-wise α optimization
    smoothing_values = np.arange(0.05, 0.6, 0.05)
    logger.info(f"Testing α values: {smoothing_values}")

    best_alpha_results = get_best_alpha_per_region(resampled, smoothing_values)
    logger.info("✅ Best EWMA α per region calculated")
    logger.info(best_alpha_results.head())

    # 7️⃣ Apply best α for each region
    smoothed_dfs = []
    for _, row in best_alpha_results.iterrows():
        region_id = row["region"]
        alpha = row["best_alpha"]
        region_df = resampled[resampled["region"] == region_id].copy()
        region_df["avg_pickups"] = region_df["total_pickups"].ewm(alpha=alpha, adjust=False).mean()
        smoothed_dfs.append(region_df)

    final_df = pd.concat(smoothed_dfs, axis=0).sort_index()
    logger.info("✅ Region-wise EWMA smoothing applied successfully")

    # 8️⃣ Save final processed dataset
    save_path = root_path / "data/processed/resampled_data.csv"
    final_df.to_csv(save_path, index=True)
    logger.info(f"Data saved successfully to {save_path}")
