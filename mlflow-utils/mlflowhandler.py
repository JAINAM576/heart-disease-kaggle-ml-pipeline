
import mlflow
import mlflow.sklearn

def log_experiment(model_name, cv_score, mean_matrics,params):

    mlflow.set_experiment("heart_disease")

    with mlflow.start_run(run_name=model_name):

        mlflow.log_params(params)
        mlflow.log_metric("cv_auc", cv_score)
        for metric_name, metric_value in mean_matrics.items():
            mlflow.log_metric(metric_name, metric_value)

        print("Logged to MLflow")
