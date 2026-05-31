import os
import mlflow
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

EXPERIMENT_NAME = "profile-deduplication"
CANDIDATE_RUN_NAME = "lightgbm_candidate"

REGISTERED_MODEL_NAME = "profile_deduplication_candidate_lightgbm"
PRODUCTION_ALIAS = "production"


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


def get_latest_model_version(
    client: MlflowClient,
    registered_model_name: str,
):
    versions = client.search_model_versions(
        filter_string=f"name='{registered_model_name}'"
    )

    if not versions:
        raise ValueError(
            f"No model versions found for registered model: "
            f"{registered_model_name}"
        )

    versions = sorted(
        versions,
        key=lambda version: int(version.version),
        reverse=True,
    )

    return versions[0]


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

print("1. Loading experiment")

experiment_id = get_experiment_id(
    client=client,
    experiment_name=EXPERIMENT_NAME,
)

print(f"Experiment ID: {experiment_id}")

print("\n2. Loading latest candidate run")

candidate_run = get_latest_run_by_name(
    client=client,
    experiment_id=experiment_id,
    run_name=CANDIDATE_RUN_NAME,
)

candidate_run_id = candidate_run.info.run_id

print(f"Candidate run_id: {candidate_run_id}")

print("\n3. Loading latest registered model version")

model_version = get_latest_model_version(
    client=client,
    registered_model_name=REGISTERED_MODEL_NAME,
)

print(f"Registered model: {REGISTERED_MODEL_NAME}")
print(f"Version: {model_version.version}")
print(f"Run ID: {model_version.run_id}")

if model_version.run_id != candidate_run_id:
    print()
    print("WARNING:")
    print("Latest model version run_id differs from latest candidate run_id.")
    print("Promotion will still use the latest registered model version.")

print("\n4. Setting production alias")

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias=PRODUCTION_ALIAS,
    version=model_version.version,
)

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=model_version.version,
    key="deployment_status",
    value="production",
)

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=model_version.version,
    key="promoted_by",
    value="quality_gate",
)

print()
print("PROMOTION COMPLETED")
print(f"Model '{REGISTERED_MODEL_NAME}'")
print(f"Version {model_version.version}")
print(f"Alias: {PRODUCTION_ALIAS}")