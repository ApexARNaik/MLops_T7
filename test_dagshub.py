
import dagshub
import mlflow

# 1. Connect to your DagsHub repository.
dagshub.init(
    repo_owner="ApexARNaik",
    repo_name="MLops_T7",
    mlflow=True,
    dvc=False
)

# 2. Display the configured tracking server.
print("MLflow Tracking URI:")
print(mlflow.get_tracking_uri())

# 3. Create an experiment.
mlflow.set_experiment("connection_test")

# 4. Log a test run.
with mlflow.start_run(
    run_name="dagshub_connection_check"
):

    mlflow.log_param("project", "AI Study Assistant")
    mlflow.log_param("owner", "AT")
    mlflow.log_param("model", "test_model")

    mlflow.log_metric("test_accuracy", 0.85)
    mlflow.log_metric("test_loss", 0.15)

    print("Run ID:", mlflow.active_run().info.run_id)

print("\nDagsHub MLflow test completed.")
