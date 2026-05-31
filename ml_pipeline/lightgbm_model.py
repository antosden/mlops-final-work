import os

from pathlib import Path

import joblib
import mlflow
import mlflow.lightgbm
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


TRAIN_PATH = Path("offline_feature_store/features/train_pairs.parquet")
VALIDATION_PATH = Path("offline_feature_store/features/validation_pairs.parquet")
TEST_PATH = Path("offline_feature_store/features/test_pairs.parquet")

MODEL_DIR = Path("model_artifacts")
MODEL_PATH = MODEL_DIR / "lightgbm_model.joblib"

EXPERIMENT_NAME = "profile-deduplication"
REGISTERED_MODEL_NAME = "lightgbm"

RANDOM_STATE = 42
THRESHOLD = 0.5
MIN_PRECISION = 0.90


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "label",
        "profile1",
        "profile2",
    }

    return [
        column for column in df.columns
        if column not in excluded_columns
    ]


def calculate_metrics(y_true, y_pred, y_pred_proba, prefix: str) -> dict:
    return {
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_pred_proba),
    }


def print_metrics(metrics: dict) -> None:
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


print("1. Reading processed datasets")

train_df = pd.read_parquet(TRAIN_PATH)
validation_df = pd.read_parquet(VALIDATION_PATH)
test_df = pd.read_parquet(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {validation_df.shape}")
print(f"Test shape: {test_df.shape}")

print("\n2. Preparing features")

feature_columns = get_feature_columns(train_df)

X_train = train_df[feature_columns].fillna(-1)
y_train = train_df["label"]

X_validation = validation_df[feature_columns].fillna(-1)
y_validation = validation_df["label"]

X_test = test_df[feature_columns].fillna(-1)
y_test = test_df["label"]

print("Features:")
for column in feature_columns:
    print(f"- {column}")

print(f"\nX_train shape: {X_train.shape}")
print(f"X_validation shape: {X_validation.shape}")
print(f"X_test shape: {X_test.shape}")

print("\n3. Training candidate LightGBM model")

model = LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    reg_lambda=1.0,
    reg_alpha=0.5,
    class_weight={0: 1.0, 1: 2.0},
    verbose=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
)

print("\n4. Predicting")

y_train_pred_proba = model.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_pred_proba >= THRESHOLD).astype(int)

y_validation_pred_proba = model.predict_proba(X_validation)[:, 1]
y_validation_pred = (y_validation_pred_proba >= THRESHOLD).astype(int)

y_test_pred_proba = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_proba >= THRESHOLD).astype(int)

print("\n5. Calculating metrics")

train_metrics = calculate_metrics(
    y_train,
    y_train_pred,
    y_train_pred_proba,
    prefix="train",
)

validation_metrics = calculate_metrics(
    y_validation,
    y_validation_pred,
    y_validation_pred_proba,
    prefix="validation",
)

test_metrics = calculate_metrics(
    y_test,
    y_test_pred,
    y_test_pred_proba,
    prefix="test",
)

print("\nTrain metrics:")
print_metrics(train_metrics)

print("\nValidation metrics:")
print_metrics(validation_metrics)

print("\nTest metrics:")
print_metrics(test_metrics)

print("\nValidation confusion matrix:")
print(confusion_matrix(y_validation, y_validation_pred))

print("\nTest confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\n6. Quality gate check")

validation_precision = validation_metrics["validation_precision"]
validation_f1 = validation_metrics["validation_f1"]

quality_gate_passed = validation_precision >= MIN_PRECISION

if quality_gate_passed:
    print("QUALITY GATE PASSED")
else:
    print("QUALITY GATE FAILED")

print(f"Validation precision: {validation_precision:.4f}")
print(f"Required precision: {MIN_PRECISION:.4f}")
print(f"Validation F1: {validation_f1:.4f}")

print("\n7. Saving candidate model")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"Saved model to: {MODEL_PATH}")

print("\n8. Logging candidate experiment to MLflow")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="lightgbm") as run:
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("role", "candidate")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("reg_lambda", 1.0)
    mlflow.log_param("reg_alpha", 0.5)
    mlflow.log_param("class_weight_0", 1.0)
    mlflow.log_param("class_weight_1", 2.0)
    mlflow.log_param("threshold", THRESHOLD)
    mlflow.log_param("min_precision", MIN_PRECISION)
    mlflow.log_param("quality_gate_passed", quality_gate_passed)
    mlflow.log_param("feature_count", len(feature_columns))

    for column in feature_columns:
        mlflow.log_param(f"feature_{column}", column)

    all_metrics = {
        **train_metrics,
        **validation_metrics,
        **test_metrics,
    }

    for metric_name, metric_value in all_metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    mlflow.log_artifact(str(MODEL_PATH))

    mlflow.lightgbm.log_model(
        lgb_model=model,
        name="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    print(f"MLflow run_id: {run.info.run_id}")

print("\nDone.")