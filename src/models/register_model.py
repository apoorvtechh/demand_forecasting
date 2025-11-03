import os
import json
import joblib
import mlflow
import logging
from pathlib import Path
from dotenv import load_dotenv
from mlflow import MlflowClient

# =========================================================
# 🌍 1️⃣ Load Environment Variables
# =========================================================
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

# =========================================================
# 🧠 2️⃣ Configure MLflow
# =========================================================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("hybrid_forecasting_pipeline")

# =========================================================
# 🧾 3️⃣ Setup Logging
# =========================================================
logger = logging.getLogger("model_registration")
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

# =========================================================
# 🧩 4️⃣ Helper: Register one model
# =========================================================
def register_single_model(model_path: Path, model_name: str, region_id: int):
    """
    Registers a single model (Prophet or XGBoost) to MLflow,
    uploading its artifact to S3.
    """
    try:
        with mlflow.start_run(run_name=f"register_{model_name}_region_{region_id}") as run:
            logger.info(f"🏃 Started MLflow run for {model_name}_region_{region_id}")

            # Load model from disk
            model = joblib.load(model_path)

            # Log model as an MLflow artifact (sent to S3 automatically)
            mlflow.log_artifact(str(model_path))

            # Log the model itself as a MLflow model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                registered_model_name=f"{model_name}_region_{region_id}"
            )

            logger.info(f"✅ Registered {model_name}_region_{region_id} successfully.")

    except Exception as e:
        logger.error(f"❌ Error registering {model_name}_region_{region_id}: {e}")
        raise


# =========================================================
# 🚀 5️⃣ Main Registration Loop
# =========================================================
if __name__ == "__main__":
    try:
        root_path = Path(__file__).parent.parent.parent
        models_dir = root_path / "models" / "training"

        # Collect all Prophet and XGBoost models
        prophet_models = sorted(models_dir.glob("prophet_region_*.joblib"))
        xgb_models = sorted(models_dir.glob("xgb_region_*.joblib"))

        if not prophet_models or not xgb_models:
            logger.warning("⚠️ No models found to register in models/training/")
            exit(0)

        logger.info(f"🧠 Found {len(prophet_models)} Prophet and {len(xgb_models)} XGBoost models for registration.")

        # Iterate through both types
        for p_model, x_model in zip(prophet_models, xgb_models):
            region_id = int(p_model.stem.split("_")[-1])

            # Prophet model
            register_single_model(p_model, "prophet", region_id)

            # XGBoost model
            register_single_model(x_model, "xgboost", region_id)

        logger.info("🏁 All regional models registered successfully to MLflow.")

    except Exception as e:
        logger.error(f"💥 Failed during registration pipeline: {e}")
        print(f"Error: {e}")
