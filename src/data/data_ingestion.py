import dask.dataframe as dd
import logging
import shutil
import tempfile
import time
from pathlib import Path

# -------------------- LOGGER SETUP --------------------
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# -------------------- CONSTANTS --------------------
MIN_LAT, MAX_LAT = 40.60, 40.85
MIN_LON, MAX_LON = -74.05, -73.70
MIN_FARE, MAX_FARE = 0.50, 81.0
MIN_DIST, MAX_DIST = 0.25, 24.43

# -------------------- FUNCTIONS --------------------
def safe_read_csv(data_path: Path, parse_dates=None, columns=None):
    """Safely read CSV file; copies to temp folder if locked (Windows-safe)."""
    try:
        return dd.read_csv(
            str(data_path),
            parse_dates=parse_dates or ["tpep_pickup_datetime"],
            usecols=columns or [
                "trip_distance", "tpep_pickup_datetime",
                "pickup_longitude", "pickup_latitude",
                "dropoff_longitude", "dropoff_latitude",
                "fare_amount",
            ],
            assume_missing=True,
            blocksize=None,
            encoding="utf-8",
        )
    except PermissionError:
        logger.warning(f"⚠️ File locked: {data_path}, copying to temp...")
        temp_file = Path(tempfile.gettempdir()) / data_path.name
        shutil.copy(data_path, temp_file)
        return dd.read_csv(str(temp_file), assume_missing=True, blocksize=None)

def clean_dask_dataframe(df):
    """Remove outliers and drop unused columns."""
    logger.info("🧹 Cleaning data...")

    df = df.loc[
        (df["pickup_latitude"].between(MIN_LAT, MAX_LAT))
        & (df["pickup_longitude"].between(MIN_LON, MAX_LON))
        & (df["dropoff_latitude"].between(MIN_LAT, MAX_LAT))
        & (df["dropoff_longitude"].between(MIN_LON, MAX_LON))
        & (df["fare_amount"].between(MIN_FARE, MAX_FARE))
        & (df["trip_distance"].between(MIN_DIST, MAX_DIST))
    ]
    df = df.drop(
        ["trip_distance", "dropoff_longitude", "dropoff_latitude", "fare_amount"],
        axis=1,
    )

    logger.info("✅ Cleaning complete.")
    return df.compute()

# -------------------- MAIN PIPELINE --------------------
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    interim_dir = root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    csvs = [
        "yellow_tripdata_2016-01.csv",
        "yellow_tripdata_2016-02.csv",
        "yellow_tripdata_2016-03.csv",
    ]

    dfs = [safe_read_csv(raw_dir / c) for c in csvs]
    df = dd.concat(dfs, axis=0)
    cleaned = clean_dask_dataframe(df)

    temp_path = interim_dir / "df_cleaned_temp.parquet"
    final_path = interim_dir / "df_cleaned.parquet"

    cleaned.to_parquet(temp_path, index=False)
    shutil.move(temp_path, final_path)
    logger.info(f"💾 Cleaned data saved to: {final_path}")
    time.sleep(1)
    logger.info("🚀 Data ingestion and cleaning pipeline completed successfully.")