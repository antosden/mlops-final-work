import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)
EXPERIMENT_NAME = "profile-deduplication"

BASELINE_RUN_NAME = "baseline_catboost"
CANDIDATE_RUN_NAME = "lightgbm_candidate"

MIN_PRECISION = 0.90
MIN_F1_DELTA = 0.00


def get_experiment_id(client: MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    return experiment.experiment_id


def get_latest_run_by_name(
    client: MlflowClient,
    experiment_id: str,
    run_name: str,
):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"Run not found: {run_name}")

    return runs[0]


def get_metric(run, metric_name: str) -> float:
    if metric_name not in run.data.metrics:
        raise ValueError(
            f"Metric '{metric_name}' not found in run {run.info.run_id}"
        )

    return run.data.metrics[metric_name]


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

print("1. Loading experiment")

experiment_id = get_experiment_id(client, EXPERIMENT_NAME)

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Experiment ID: {experiment_id}")

print("\n2. Loading latest runs")

baseline_run = get_latest_run_by_name(
    client=client,
    experiment_id=experiment_id,
    run_name=BASELINE_RUN_NAME,
)

candidate_run = get_latest_run_by_name(
    client=client,
    experiment_id=experiment_id,
    run_name=CANDIDATE_RUN_NAME,
)

print(f"Baseline run_id: {baseline_run.info.run_id}")
print(f"Candidate run_id: {candidate_run.info.run_id}")

print("\n3. Reading metrics")

baseline_validation_precision = get_metric(
    baseline_run,
    "validation_precision",
)
baseline_validation_f1 = get_metric(
    baseline_run,
    "validation_f1",
)
baseline_test_precision = get_metric(
    baseline_run,
    "test_precision",
)
baseline_test_f1 = get_metric(
    baseline_run,
    "test_f1",
)

candidate_validation_precision = get_metric(
    candidate_run,
    "validation_precision",
)
candidate_validation_f1 = get_metric(
    candidate_run,
    "validation_f1",
)
candidate_test_precision = get_metric(
    candidate_run,
    "test_precision",
)
candidate_test_f1 = get_metric(
    candidate_run,
    "test_f1",
)

print("\nBaseline metrics:")
print(f"validation_precision: {baseline_validation_precision:.4f}")
print(f"validation_f1: {baseline_validation_f1:.4f}")
print(f"test_precision: {baseline_test_precision:.4f}")
print(f"test_f1: {baseline_test_f1:.4f}")

print("\nCandidate metrics:")
print(f"validation_precision: {candidate_validation_precision:.4f}")
print(f"validation_f1: {candidate_validation_f1:.4f}")
print(f"test_precision: {candidate_test_precision:.4f}")
print(f"test_f1: {candidate_test_f1:.4f}")

print("\n4. Quality gate")

precision_gate = candidate_validation_precision >= MIN_PRECISION
f1_gate = candidate_validation_f1 >= baseline_validation_f1 + MIN_F1_DELTA

quality_gate_passed = precision_gate and f1_gate

print(f"Precision gate: {precision_gate}")
print(
    f"Candidate validation precision: "
    f"{candidate_validation_precision:.4f}"
)
print(f"Required precision: {MIN_PRECISION:.4f}")

print()
print(f"F1 gate: {f1_gate}")
print(f"Baseline validation F1: {baseline_validation_f1:.4f}")
print(f"Candidate validation F1: {candidate_validation_f1:.4f}")
print(f"Required F1 delta: {MIN_F1_DELTA:.4f}")

print("\n5. Decision")

if quality_gate_passed:
    print("DECISION: PROMOTE_CANDIDATE")
    print("Candidate model passed quality gate.")
    sys.exit(0)

print("DECISION: REJECT_CANDIDATE")
print("Candidate model failed quality gate.")
sys.exit(1)