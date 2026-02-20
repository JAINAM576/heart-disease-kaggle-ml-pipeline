import mlflow
import mlflow.sklearn
import joblib
import os
import tempfile
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


def register_model_from_run(
    tracking_uri: str,
    run_id: str,
    artifact_path: str,
    registered_model_name: str,
    experiment_name: str = None,
    stage: str = None 
):
 
    load_dotenv()

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    artifact_uri = f"runs:/{run_id}/{artifact_path}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=tmp_dir
        )
        model = joblib.load(local_path)

    with mlflow.start_run() as run:
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )

    print(f"✅ Model registered as version: {model_info.registered_model_version}")

    # 🔥 Transition stage ONLY if provided
    if stage:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_info.registered_model_version,
            stage=stage
        )
        print(f"🚀 Model moved to stage: {stage}")

    print("✅ Done.")


if __name__ == "__main__":
    register_model_from_run(
        tracking_uri="https://dagshub.com/JAINAM576/heart-disease-kaggle-ml-pipeline.mlflow",
        run_id="6d81cd4b758d45c2a812c66c93910317",
        artifact_path="stacking_model.pkl",
        registered_model_name="stacking_heart_disease_model1",
        experiment_name="heart_disease_stacking",
        # stage="Production"
    )