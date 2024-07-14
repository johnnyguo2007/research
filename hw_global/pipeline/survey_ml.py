import mlflow
from mlflow.entities import ViewType
import pandas as pd

# Set the MLflow tracking URI to the same one used in the experiments
mlflow.set_tracking_uri("http://127.0.0.1:8080")


def get_latest_run_metrics(experiment_name):
    """
    Retrieve the day_model_val_rmse metric from the latest run of the given experiment.
    """
    # Get the experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None

    # Get all runs for this experiment, sorted by start time
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
        run_view_type=ViewType.ACTIVE_ONLY
    )

    if runs.empty:
        print(f"No runs found for experiment '{experiment_name}'.")
        return None

    # Get the latest run
    latest_run = runs.iloc[0]

    # Extract the day_model_val_rmse metric
    day_model_val_rmse = latest_run.get("metrics.day_model_val_rmse")

    return {
        "experiment_name": experiment_name,
        "run_id": latest_run.run_id,
        "day_model_val_rmse": day_model_val_rmse
    }


def main():
    # Get all experiments
    experiments = mlflow.search_experiments()

    results = []

    for exp in experiments:
        if exp.name.startswith("Year"):
                # or exp.name.startswith("production_Day_")):
            result = get_latest_run_metrics(exp.name)
            if result:
                results.append(result)

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Sort the DataFrame by day_model_val_rmse
    df_sorted = df.sort_values("day_model_val_rmse")

    # Save the results to a CSV file
    output_path = "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary/Year_val_rmse_results.csv"
    df_sorted.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Display the results
    print(df_sorted)


if __name__ == "__main__":
    main()