import os
import json
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve
)
import pandas as pd

train_df=pd.read_csv(r'C:\Users\HP\Documents\Kaggle_Compition_Bacancy\data\raw\train.csv')
train_df["Heart Disease"]=train_df["Heart Disease"].map({"Presence":1,"Absence":0})
train_df = train_df.drop(columns=["id"])
Y=train_df.iloc[:,-1]

import dagshub
dagshub.init(repo_owner='JAINAM576', repo_name='heart-disease-kaggle-ml-pipeline', mlflow=True)


def log_from_local_directory(
    base_dir,
    model_type,
    Y_true,
    threshold=0.5,
):
    """
    Automatically scans local directory and logs everything to MLflow.
    
    Expected files inside base_dir:
        - best_params_<model_type>.json
        - oof_preds_<model_type>.npy
        - test_preds_<model_type>.npy
        - <model_type>_fold_*.pkl
    """

    print(f"🔍 Scanning directory: {base_dir}")

    # ============================
    # Locate Files
    # ============================
    best_params_path = None
    oof_path = None
    test_path = None
    fold_models = []

    for root, _, files in os.walk(base_dir):
        for file in files:

            full_path = os.path.join(root, file)

            if file == f"best_params_{model_type}.json":
                best_params_path = full_path

            elif file == f"oof_preds_{model_type}.npy":
                oof_path = full_path

            elif file == f"test_preds_{model_type}.npy":
                test_path = full_path

            elif file.startswith(f"{model_type}_fold_") and file.endswith(".pkl"):
                fold_models.append(full_path)

    # ============================
    # Load Files
    # ============================
    best_params = {}
    if best_params_path:
        with open(best_params_path, "r") as f:
            best_params = json.load(f)

    if not oof_path:
        raise ValueError("OOF predictions not found!")

    oof_preds = np.load(oof_path)

    # ============================
    # Compute Metrics
    # ============================
    y_pred_labels = (oof_preds > threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(Y_true, oof_preds),
        "accuracy": accuracy_score(Y_true, y_pred_labels),
        "precision": precision_score(Y_true, y_pred_labels),
        "recall": recall_score(Y_true, y_pred_labels),
        "f1_score": f1_score(Y_true, y_pred_labels),
        "log_loss": log_loss(Y_true, oof_preds),
    }

    # ============================
    # MLflow Logging
    # ============================
    mlflow.set_experiment(f"{model_type}_experiment_best_params_optuna")

    with mlflow.start_run(run_name=f"{model_type}_local_logged_run"):

        # Log params
        if best_params:
            mlflow.log_params(best_params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Threshold analysis
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds = (oof_preds > t).astype(int)
            mlflow.log_metric(f"f1_at_{t}", f1_score(Y_true, preds))

        # Log confusion matrix
        cm = confusion_matrix(Y_true, y_pred_labels)
        np.save("confusion_matrix.npy", cm)
        mlflow.log_artifact("confusion_matrix.npy")

        # Log classification report
        report = classification_report(Y_true, y_pred_labels)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Log ROC curve
        fpr, tpr, _ = roc_curve(Y_true, oof_preds)
        roc_data = np.vstack((fpr, tpr)).T
        np.save("roc_curve.npy", roc_data)
        mlflow.log_artifact("roc_curve.npy")

        # Log stored artifacts
        mlflow.log_artifact(oof_path)

        if test_path:
            mlflow.log_artifact(test_path)

        for model_path in fold_models:
            mlflow.log_artifact(model_path)

        # Log one model example if exists
        if fold_models:
            example_model = joblib.load(fold_models[0])
            mlflow.sklearn.log_model(example_model, "example_model")

        mlflow.log_metric("num_folds_detected", len(fold_models))
        mlflow.log_metric("num_samples", len(Y_true))

    print("✅ Local files successfully logged to MLflow")

log_from_local_directory(
    base_dir=r"C:\Users\HP\Documents\Kaggle_Compition_Bacancy\models\XgBoost",
    model_type="xgb",
    Y_true=Y
)


# import mlflow
# import numpy as np
# import json
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# from pathlib import Path
# from sklearn.metrics import ConfusionMatrixDisplay

# # ── Config ────────────────────────────────────────────────────────────────────
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_experiment("stacking_experiment_histgradient")

# stack_logs = Path(r"C:\Users\HP\Documents\Kaggle_Compition_Bacancy\notebooks\stacking_logs")

# # ── Helper: Parse key-value text files ───────────────────────────────────────
# def parse_txt_to_dict(filepath):
#     """Parse lines like 'key: value' or 'key = value' into a dict."""
#     result = {}
#     with open(filepath) as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue
#             for sep in [":", "="]:
#                 if sep in line:
#                     k, v = line.split(sep, 1)
#                     k = k.strip().replace(" ", "_").lower()
#                     v = v.strip()
#                     try:
#                         result[k] = float(v)
#                     except ValueError:
#                         result[k] = v   # keep as string if not numeric
#                     break
#     return result

# # ── Helper: Confusion matrix plot ─────────────────────────────────────────────
# def save_confusion_matrix_plot(npy_path, out_path):
#     cm = np.load(npy_path)
#     fig, ax = plt.subplots(figsize=(6, 5))
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(ax=ax, colorbar=True, cmap="Blues")
#     ax.set_title("Confusion Matrix\nMeta: HistGradientBoosting  |  Base: LGB + XGB + CAT")
#     plt.tight_layout()
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)
#     print(f"  ✅ Confusion matrix plot saved: {out_path.name}")

# # ── Helper: ROC curve plot ────────────────────────────────────────────────────
# def save_roc_curve_plot(npy_path, out_path, auc_score=None):
#     data = np.load(npy_path, allow_pickle=True)

#     # Handle different save formats
#     if data.ndim == 0:              # saved as dict object
#         d = data.item()
#         fpr, tpr = d["fpr"], d["tpr"]
#     elif data.ndim == 2:            # shape (2, N) → [fpr, tpr]
#         fpr, tpr = data[0], data[1]
#     else:
#         print("  ⚠️  Unexpected roc_curve.npy format — skipping plot")
#         return

#     fig, ax = plt.subplots(figsize=(6, 5))
#     label = f"ROC AUC = {auc_score:.4f}" if auc_score else "ROC Curve"
#     ax.plot(fpr, tpr, color="darkorange", lw=2, label=label)
#     ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.set_title("ROC Curve\nMeta: HistGradientBoosting  |  Base: LGB + XGB + CAT")
#     ax.legend(loc="lower right")
#     plt.tight_layout()
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)
#     print(f"  ✅ ROC curve plot saved: {out_path.name}")

# # ── Start MLflow Run ──────────────────────────────────────────────────────────
# with mlflow.start_run(run_name="hist_meta_stacking"):

#     # 1. Tags — architecture description
#     mlflow.set_tags({
#         "meta_model":    "HistGradientBoostingClassifier",
#         "base_model_1":  "LightGBM",
#         "base_model_2":  "XGBoost",
#         "base_model_3":  "CatBoost",
#         "stacking_type": "OOF Stacking",
#         "task":          "Binary Classification",
#         "description":   "2-level stacking: LGB + XGB + CAT  →  HistGBM meta",
#     })
#     print("✅ Tags logged")

#     # 2. Meta model params → log as MLflow params
#     meta_params_file = stack_logs / "meta_model_params.txt"
#     if meta_params_file.exists():
#         params = parse_txt_to_dict(meta_params_file)
#         # MLflow params must be strings
#         mlflow.log_params({k: str(v) for k, v in params.items()})
#         print(f"✅ Meta model params logged ({len(params)} params)")

#     # 3. Metrics → log as MLflow metrics
#     metrics_file = stack_logs / "metrics.txt"
#     if metrics_file.exists():
#         metrics = parse_txt_to_dict(metrics_file)
#         numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
#         mlflow.log_metrics(numeric_metrics)
#         print(f"✅ Metrics logged ({len(numeric_metrics)} values): {list(numeric_metrics.keys())}")

#     # 4. Threshold F1 → log as MLflow metrics
#     threshold_file = stack_logs / "threshold_f1.txt"
#     if threshold_file.exists():
#         t_data = parse_txt_to_dict(threshold_file)
#         numeric_t = {k: v for k, v in t_data.items() if isinstance(v, float)}
#         mlflow.log_metrics(numeric_t)
#         print(f"✅ Threshold/F1 metrics logged: {numeric_t}")

#     # 5. Confusion matrix → plot + artifact
#     cm_npy = stack_logs / "confusion_matrix.npy"
#     cm_png = stack_logs / "confusion_matrix.png"
#     if cm_npy.exists():
#         save_confusion_matrix_plot(cm_npy, cm_png)
#         mlflow.log_artifact(str(cm_png), artifact_path="plots")

#     # Also log confusion_matrix.txt if present
#     cm_txt = stack_logs / "confusion_matrix.txt"
#     if cm_txt.exists():
#         mlflow.log_artifact(str(cm_txt), artifact_path="reports")

#     # 6. ROC curve → plot + artifact
#     roc_npy = stack_logs / "roc_curve.npy"
#     roc_png = stack_logs / "roc_curve.png"
#     if roc_npy.exists():
#         # Pull AUC from already-parsed metrics if available
#         auc = numeric_metrics.get("roc_auc") or numeric_metrics.get("auc")
#         save_roc_curve_plot(roc_npy, roc_png, auc_score=auc)
#         mlflow.log_artifact(str(roc_png), artifact_path="plots")

#     # 7. Classification report → artifact
#     clf_report = stack_logs / "classification_report.txt"
#     if clf_report.exists():
#         mlflow.log_artifact(str(clf_report), artifact_path="reports")
#         print("✅ Classification report logged")

#     # 8. All raw .npy arrays → artifacts
#     for npy_file in stack_logs.glob("*.npy"):
#         mlflow.log_artifact(str(npy_file), artifact_path="arrays")
#     print("✅ All .npy arrays logged")

#     # 9. All .txt files → artifacts (as backup/reference)
#     for txt_file in stack_logs.glob("*.txt"):
#         mlflow.log_artifact(str(txt_file), artifact_path="reports")
#     print("✅ All .txt reports logged")

# print("\n🎉 MLflow run complete! Everything logged from stacking_logs/")