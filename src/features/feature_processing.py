import logging
from pathlib import Path
import pandas as pd

# ---------------------------
# 📜 Logger Configuration
# ---------------------------
logger = logging.getLogger("feature_processing")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# ---------------------------
# ⚙️ Main Execution
# ---------------------------
if __name__ == "__main__":
    # Define paths
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    data_path = root_path / "data/processed/resampled_data.csv"

    # Read input data
    df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("✅ Data read successfully")

    # ---------------------------
    # 🧭 Datetime-based features
    # ---------------------------
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_of_week
    df["month"] = df["tpep_pickup_datetime"].dt.month
    logger.info("🕓 Datetime features extracted successfully")

    # Set datetime as index for time-series operations
    df.set_index("tpep_pickup_datetime", inplace=True)
    logger.info("📅 Datetime column set as index successfully")

    # ---------------------------
    # 🔁 Region-wise Feature Engineering
    # ---------------------------
    region_grp = df.groupby("region")

    # ✅ 1️⃣ Lag features (previous time steps)
    periods = list(range(1, 5))  # lag_1 to lag_4
    lag_features = region_grp["total_pickups"].shift(periods)
    logger.info("🧩 Lag features generated successfully")

    # ✅ 2️⃣ Rolling statistics
    df["rolling_mean_3"] = region_grp["total_pickups"].transform(lambda x: x.rolling(window=3).mean())
    df["rolling_std_3"]  = region_grp["total_pickups"].transform(lambda x: x.rolling(window=3).std())
    df["rolling_mean_6"] = region_grp["total_pickups"].transform(lambda x: x.rolling(window=6).mean())
    df["rolling_std_6"]  = region_grp["total_pickups"].transform(lambda x: x.rolling(window=6).std())
    logger.info("📈 Rolling window statistics added successfully")

    # ---------------------------
    # 🧩 Combine lag features with main df
    # ---------------------------
    data = pd.concat([lag_features, df], axis=1)
    logger.info("🔗 Lag features merged successfully")

    # Drop missing values (due to lag/rolling windows)
    data.dropna(inplace=True)
    logger.info("🧹 Missing values dropped successfully")

    # Rename lag columns
    mapper = {name: f"lag_{i+1}" for i, name in enumerate(data.columns[0:4])}
    data.rename(columns=mapper, inplace=True)
    logger.info("📝 Column names renamed successfully")

    # ---------------------------
    # ✂️ Split train/test
    # ---------------------------
    trainset = data[data["month"].isin([1, 2])]
    testset = data[data["month"].isin([3])]
    logger.info(f"📊 Data split into Train ({len(trainset)}) and Test ({len(testset)})")

    # ---------------------------
    # 💾 Save processed datasets
    # ---------------------------
    train_path = root_path / "data/processed/train.csv"
    test_path = root_path / "data/processed/test.csv"

    trainset.to_csv(train_path, index=True)
    testset.to_csv(test_path, index=True)

    logger.info("💾 Train and Test datasets saved successfully")
    logger.info("✅ Feature processing stage completed!")
