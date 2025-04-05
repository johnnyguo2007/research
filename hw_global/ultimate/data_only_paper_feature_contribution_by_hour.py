import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
import mlflow
from typing import List, Dict, Tuple, Optional
from mlflow_tools import (
    create_day_night_summary,
    get_feature_groups,
    generate_group_shap_plots_by_climate_zone, # Keep for data calculation logic inside
)
from mlflow_tools.group_data import calculate_group_shap_values, GroupData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Copied from mlflow_tools.plot_shap_stacked_bar
def _save_data_csv(
    df: pd.DataFrame, total_values: pd.Series, output_path: str, data_type: str
) -> None:
    """
    Save data to CSV files, including both individual values and totals.

    Args:
        df: DataFrame containing the data
        total_values: Series containing the total values
        output_path: Base path for the output file (e.g., .../Arid/FGR/shap_and_feature_values_FGR_Arid) - NO extension expected.
        data_type: String indicating the type of data ('shap' or 'feature' or 'group')
    """
    # The output_path provided IS the base path, no need to rsplit.
    base_path = output_path
    csv_path = f"{base_path}_{data_type}_data.csv"

    # Create a copy of the DataFrame
    output_df = df.copy()

    # Add the total values as a new column if it's not empty
    if not total_values.empty:
        output_df["Total"] = total_values

    # Save to CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    output_df.to_csv(csv_path)

    logging.info(f"Saved {data_type} data to {csv_path}")


def get_experiment_and_run(experiment_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieves the experiment and the latest run from MLflow.

    Args:
        experiment_name (str): Name of the MLflow experiment.

    Returns:
        tuple: The experiment ID and the latest run ID.
    """
    mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")
    logging.info("Set MLflow tracking URI")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.error(f"Experiment '{experiment_name}' not found.")
        return None, None

    experiment_id = experiment.experiment_id
    logging.info(f"Found experiment with ID: {experiment_id}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1
    )
    if len(runs) == 0:
        logging.error(f"No runs found in experiment '{experiment_name}'.")
        return experiment_id, None

    run = runs.iloc[0]
    run_id = run.run_id
    logging.info(f"Processing latest run with ID: {run_id}")

    return experiment_id, run_id

def extract_shap_and_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Extracts SHAP and corresponding feature columns from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing SHAP values.

    Returns:
        tuple: (list of SHAP columns, list of feature columns)
    """
    shap_cols = [col for col in df.columns if col.endswith("_shap")]
    feature_cols = [col.replace("_shap", "") for col in shap_cols]
    return shap_cols, feature_cols


def save_hourly_group_data(
    obj_group_data: GroupData,
    kg_class: str,
    output_dir: str,
    base_values: pd.Series
) -> None:
    """Calculates and saves the hourly mean group SHAP data."""
    feature_group_data = obj_group_data.shap_group_hourly_mean_df(kg_class) # This is pivot_df in plot func
    if feature_group_data.empty:
        logging.warning(f"No hourly group data for KGMajorClass '{kg_class}'. Skipping CSV saving.")
        return

    mean_values = feature_group_data.sum(axis=1) + base_values
    output_path_base = os.path.join(
        output_dir, f"feature_group_contribution_by_hour_{kg_class}" # Base name without extension
    )
    _save_data_csv(feature_group_data, mean_values, output_path_base, "group")


def save_climate_zone_group_details(
    obj_group_data: GroupData,
    kg_class: str,
    output_dir: str,
    base_values: pd.Series
) -> None:
    """Calculates and saves detailed SHAP and Feature CSVs per group within a climate zone."""
    kg_class_dir = os.path.join(output_dir, kg_class)
    os.makedirs(kg_class_dir, exist_ok=True)

    # Save the overall KG class SHAP data first
    shap_plot_df_kg = obj_group_data.shap_group_hourly_mean_df(kg_class)
    if shap_plot_df_kg.empty:
        logging.warning(f"No SHAP data for KGMajorClass '{kg_class}'. Skipping overall KG SHAP CSV.")
    else:
        mean_shap_kg = shap_plot_df_kg.sum(axis=1) + base_values # Corresponds to mean_shap in plot_shap_stacked_bar
        kg_shap_output_path_base = os.path.join(
            kg_class_dir, f"{kg_class}_shap_stacked_bar_all_features" # Base name without extension
        )
        _save_data_csv(shap_plot_df_kg, mean_shap_kg, kg_shap_output_path_base, "shap")

    # Save data for each group within the climate zone
    for group_name in obj_group_data.group_names:
        group_dir = os.path.join(kg_class_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        # Get SHAP and feature data for this group and kg_class
        # SHAP data (hourly mean for *one* group in the KG class)
        shap_df_group = obj_group_data.shap_group_hourly_mean_df(kg_class)[[group_name]]
        # Feature data (hourly mean features for *one* group in the KG class)
        feature_values_df_group = obj_group_data.feature_hourly_mean_for_a_given_group_df(kg_class, group_name)

        if shap_df_group.empty or feature_values_df_group.empty:
            logging.warning(
                f"No data available for group '{group_name}' in KGMajorClass '{kg_class}'. Skipping group detail CSVs."
            )
            continue

        # --- Data saving mimicking create_side_by_side_group_plot ---
        output_base_combined = os.path.join(group_dir, f"shap_and_feature_values_{group_name}_{kg_class}")
        # Calculate totals
        total_shap = shap_df_group.sum(axis=1) # Total SHAP for this specific group
        total_features = (
            feature_values_df_group.sum(axis=1)
            if len(feature_values_df_group.columns) > 1
            else pd.Series(dtype='float64', index=feature_values_df_group.index) # Ensure correct type and index even if empty sum
        )
        # Save SHAP data associated with the combined plot naming convention
        _save_data_csv(shap_df_group, total_shap, output_base_combined, "shap")
        # Save Feature data associated with the combined plot naming convention
        _save_data_csv(feature_values_df_group, total_features, output_base_combined, "feature")

        # --- Data saving mimicking the standalone shap_contributions plot ---
        # Note: The standalone plot in the original script uses plot_shap_stacked_bar,
        # which saves data using its own output path. Here we replicate that save.
        output_base_standalone = os.path.join(group_dir, f"shap_contributions_{group_name}_{kg_class}")
        # Calculate mean_shap for the standalone plot (total shap + base value for the group)
        # Base value is tricky here - the original plot didn't seem to use it correctly for single group plots.
        # Let's save the raw group shap total without base value for consistency with combined plot data save.
        _save_data_csv(shap_df_group, total_shap, output_base_standalone, "shap")


def main():
    """Main function to process SHAP values and generate data CSVs."""
    logging.info("Starting feature contribution data generation...")

    # Parse arguments
    args = parse_arguments()
    logging.info(f"Parsed command line arguments: {vars(args)}")

    # Set flags based on --all argument if provided
    if args.all:
        args.day_night_summary = True
        args.group_shap_data = True

    # Get experiment and run
    experiment_id, run_id = get_experiment_and_run(args.experiment_name)
    if experiment_id is None or run_id is None:
        return

    # Setup paths
    mlflow_run = mlflow.get_run(run_id)
    artifact_uri = mlflow_run.info.artifact_uri
    # --- IMPORTANT: Adjust this path replacement based on your MLflow server setup ---
    artifact_uri = artifact_uri.replace(
        "mlflow-artifacts:",
        "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts",
    )
    # --- Check if the above replacement is correct for your environment ---
    if not os.path.isdir(artifact_uri):
         logging.warning(f"Adjusted artifact URI does not exist: {artifact_uri}. Trying original URI.")
         artifact_uri = mlflow_run.info.artifact_uri # Fallback to original
         if artifact_uri.startswith("file://"):
              artifact_uri = artifact_uri[len("file://"):] # Remove file:// prefix if present

    shap_values_feather_path = os.path.join(
        artifact_uri, "shap_values_with_additional_columns.feather"
    )
    # Use a different output directory name to avoid conflicts with plot script
    output_dir = os.path.join(artifact_uri, "data_only_24_hourly")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir}")


    if not os.path.exists(shap_values_feather_path):
        logging.error(f"SHAP values file not found at: {shap_values_feather_path}")
        return

    # Load raw data
    try:
        all_df = pd.read_feather(shap_values_feather_path)
        logging.info(f"Loaded data from {shap_values_feather_path}. Shape: {all_df.shape}")
    except Exception as e:
        logging.error(f"Failed to load feather file: {e}")
        return

    if "base_value" not in all_df.columns:
        logging.warning("'base_value' column not found. Using 0 as base value.")
        all_df['base_value'] = 0 # Add base_value if missing

    base_value_global_mean = all_df["base_value"].mean() # Used for day/night summary

    # Get feature names and groups
    shap_cols, feature_cols = extract_shap_and_feature_columns(all_df)
    if not feature_cols:
        logging.error("Could not extract feature columns.")
        return
    feature_to_group_mapping = get_feature_groups(feature_cols)

    # Calculate initial group values (needed for day/night summary)
    # Need the original all_df for create_day_night_summary context
    obj_group_data_full = calculate_group_shap_values(all_df.copy(), feature_to_group_mapping)

    logging.info(f"Initial GroupData created. Group names: {obj_group_data_full.group_names}")

    # Generate day/night summary CSV if requested
    if args.day_night_summary:
        logging.info("Generating day/night summary...")
        try:
            # Use the shap_group_detail_df which contains the summed SHAP values per group
            summary_df = create_day_night_summary(obj_group_data_full._shap_group_detail_df, all_df, output_dir)
            logging.info("\nFeature Group Day/Night Summary:")
            logging.info("\n" + str(summary_df))
        except Exception as e:
            logging.error(f"Error generating day/night summary: {e}")


    # --- Hourly Aggregation and Data Saving ---
    if args.group_shap_data:
        logging.info("Processing and saving hourly group SHAP data...")
        # Base values per hour
        base_values_hourly: pd.Series = all_df.groupby("local_hour")["base_value"].first()

        # Group all_df by local_hour and KGMajorClass, calculate mean
        exclude_cols = ["global_event_ID", "lon", "lat", "time", "KGClass"]
        cols_to_drop = [col for col in exclude_cols if col in all_df.columns]
        all_df_by_hour_kg = all_df.drop(columns=cols_to_drop).groupby(["local_hour", "KGMajorClass"], observed=True).mean().reset_index()

        if all_df_by_hour_kg.empty:
             logging.error("Grouping data by hour and KG class resulted in an empty DataFrame. Cannot proceed.")
             return

        # Calculate group values for the hourly aggregated data
        obj_group_data_hourly = calculate_group_shap_values(all_df_by_hour_kg, feature_to_group_mapping)
        logging.info("Hourly GroupData created.")


        # Generate data CSVs for each climate zone
        kg_classes = ["global"] + all_df_by_hour_kg["KGMajorClass"].dropna().unique().tolist()

        for kg_class in kg_classes:
            logging.info(f"--- Processing data for KGMajorClass: {kg_class} ---")
            # Save hourly mean group SHAP contributions (like feature_group_contribution_by_hour_*.png data)
            save_hourly_group_data(
                obj_group_data_hourly,
                kg_class,
                output_dir,
                base_values_hourly
            )

            # Save detailed SHAP and Feature CSVs per group within the climate zone
            # (like data for shap_and_feature_values_*.png and shap_contributions_*.png)
            save_climate_zone_group_details(
                obj_group_data_hourly,
                kg_class,
                output_dir,
                base_values_hourly
            )

    logging.info("Data generation completed successfully.")
    logging.info(f"All CSV outputs have been saved to: {output_dir}")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process SHAP values and save data contributions by feature group and hour as CSV."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the MLflow experiment to process.",
    )
    # Arguments to control which data files are generated
    parser.add_argument(
        "--day-night-summary",
        action="store_true",
        default=False,
        help="Generate the day/night summary CSV.",
    )
    parser.add_argument(
        "--group-shap-data",
        action="store_true",
        default=False,
        help="Generate the hourly group SHAP and feature data CSVs by climate zone.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all data generation steps (day/night summary, hourly group data).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()
