import os
import mlflow
import joblib
import logging
from pathlib import Path
from dotenv import load_dotenv

# -------------------------------------------------
# 🌍 Load environment variables
# -------------------------------------------------
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# ✅ Ensure the experiment exists or create it
mlflow.set_experiment("hybrid_forecasting_pipeline")

# -------------------------------------------------
# 🧾 Logger setup
# -------------------------------------------------
logger = logging.getLogger("register_scaler_kmeans")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -------------------------------------------------
# 🧠 Function to log artifacts (scaler & kmeans)
# -------------------------------------------------
def log_artifact_only(name, artifact_path):
    """
    Logs an artifact (e.g., scaler.joblib or mb_kmeans.joblib) under the current MLflow experiment.
    Returns run_id and artifact_uri.
    """
    try:
        with mlflow.start_run(run_name=f"log_{name}") as run:
            mlflow.log_artifact(str(artifact_path))
            artifact_uri = mlflow.get_artifact_uri()
            logger.info(f"✅ Logged {name} at {artifact_uri}")
            logger.info(f"🧩 Run ID: {run.info.run_id}")
            return run.info.run_id, artifact_uri

    except Exception as e:
        logger.error(f"❌ Failed to log {name}: {e}")
        raise

# -------------------------------------------------
# 🚀 Main script
# -------------------------------------------------
if __name__ == "__main__":
    current_path = Path(__file__).resolve()
    root_path = current_path.parent.parent.parent
    models_dir = root_path / "models"

    scaler_path = models_dir / "scaler.joblib"
    kmeans_path = models_dir / "mb_kmeans.joblib"

    # Validate file presence
    if not scaler_path.exists() or not kmeans_path.exists():
        logger.error("⚠️ Missing scaler or kmeans files in 'models/'. Please check before running.")
        exit(1)

    logger.info("🚀 Starting MLflow artifact logging for Scaler and KMeans under 'hybrid_forecasting_pipeline'...")

    run_id_1, uri_1 = log_artifact_only("scaler_preprocessor", scaler_path)
    run_id_2, uri_2 = log_artifact_only("kmeans_cluster_model", kmeans_path)

    logger.info(f"🏁 Artifacts logged successfully under experiment: hybrid_forecasting_pipeline")
    logger.info(f"🔗 Scaler Run: {run_id_1}")
    logger.info(f"🔗 KMeans Run: {run_id_2}")
