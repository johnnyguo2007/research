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
    # generate_group_shap_plots_by_climate_zone, # No longer needed for just data
)
from mlflow_tools.plot_util import replace_cold_with_continental # Import the function
from mlflow_tools.group_data import calculate_group_shap_values, GroupData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Copied from mlflow_tools.plot_shap_stacked_bar
def _save_data_csv(
    df: pd.DataFrame, output_path: str, data_type: str, total_values: Optional[pd.Series] = None
) -> None:
    """
    Save data DataFrame to a CSV file, optionally adding a 'Total' column.

    Args:
        df: DataFrame containing the data to save.
        output_path: Base path for the output file (e.g., .../Arid/Arid) - NO extension expected.
        data_type: String suffix for the filename (e.g., 'shap', 'feature', 'group_shap_contribution').
        total_values (pd.Series, optional): Series containing total values to add as a 'Total' column. Defaults to None.
    """
    base_path = output_path
    # Construct the final CSV path using the data_type suffix
    csv_path = f"{base_path}_{data_type}_data.csv"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Create a copy to avoid modifying the original DataFrame
    output_df = df.copy()

    # Add the total values as a new column if provided
    if total_values is not None and not total_values.empty:
        output_df["Total"] = total_values

    # Save to CSV
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


def save_climate_zone_group_details(
    obj_group_data: GroupData,
    kg_class: str, # Original KG class name (e.g., 'Cold')
    output_dir: str,
    base_values: pd.Series
) -> None:
    """Calculates and saves separate CSVs for hourly mean SHAP (individual), Features (individual), and SHAP (grouped) for a climate zone, renaming 'Cold' to 'Continental' in paths/filenames."""
    
    # Get the display name for paths/filenames
    kg_class_display_name = replace_cold_with_continental(kg_class)
    
    # Create directory using the display name
    kg_class_dir = os.path.join(output_dir, kg_class_display_name) # e.g., .../Continental/
    os.makedirs(kg_class_dir, exist_ok=True)

    # --- Data for individual features --- 

    # Get data using the ORIGINAL kg_class name
    shap_individual_features_df = obj_group_data.shap_hourly_mean_df(kg_class)
    features_individual_features_df = obj_group_data.feature_hourly_mean_df(kg_class)

    # Check if individual feature dataframes are empty
    if shap_individual_features_df.empty or features_individual_features_df.empty:
        logging.warning(f"No hourly individual SHAP or Feature data available for KGMajorClass '{kg_class}'. Skipping individual feature CSVs.")
    else:
        # Define base path for these files using the DISPLAY name
        output_path_base_individual = os.path.join(kg_class_dir, f"shap_and_feature_values_{kg_class_display_name}") # e.g., .../Continental/shap_and_feature_values_Continental
        # Save individual SHAP values
        _save_data_csv(shap_individual_features_df, output_path_base_individual, "shap") # Saves as ..._shap_data.csv
        # Save individual Feature values
        _save_data_csv(features_individual_features_df, output_path_base_individual, "feature") # Saves as ..._feature_data.csv

    # --- Data for grouped features --- 

    # Get data using the ORIGINAL kg_class name
    shap_grouped_features_df = obj_group_data.shap_group_hourly_mean_df(kg_class)

    # Check if grouped dataframe is empty
    if shap_grouped_features_df.empty:
        logging.warning(f"No hourly grouped SHAP data available for KGMajorClass '{kg_class}'. Skipping grouped SHAP CSV.")
    else:
        # Define base path for this file using the DISPLAY name
        output_path_base_grouped = os.path.join(kg_class_dir, f"{kg_class_display_name}") # e.g., .../Continental/Continental
        
        # Calculate the Total column (Sum of group SHAP + Base Value for each hour)
        # Ensure base_values index aligns with shap_grouped_features_df index (local_hour)
        if not base_values.index.equals(shap_grouped_features_df.index):
             logging.warning(f"Index mismatch between base_values and grouped SHAP for {kg_class}. Cannot calculate Total reliably. Saving without Total.")
             total_values_with_base = None
        else:
             total_values_with_base = shap_grouped_features_df.sum(axis=1) + base_values

        # Save grouped SHAP values using a descriptive data_type and include the total
        _save_data_csv(shap_grouped_features_df, output_path_base_grouped, "group_shap_contribution", total_values=total_values_with_base) # Saves as ..._group_shap_contribution_data.csv


def main():
    """Main function to process SHAP values and generate data CSVs."""
    logging.info("Starting feature contribution data generation...")

    # Parse arguments
    args = parse_arguments()
    logging.info(f"Parsed command line arguments: {vars(args)}")

    # No need for flags since --all is always assumed
    # # Set flags based on --all argument if provided
    # if args.all:
    #     args.day_night_summary = True
    #     args.group_shap_data = True

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

    # --- Always generate day/night summary --- 
    # if args.day_night_summary:
    logging.info("Generating day/night summary...")
    try:
        # Use the shap_group_detail_df which contains the summed SHAP values per group
        summary_df = create_day_night_summary(obj_group_data_full._shap_group_detail_df, all_df, output_dir)
        logging.info("\nFeature Group Day/Night Summary:")
        logging.info("\n" + str(summary_df))
    except Exception as e:
        logging.error(f"Error generating day/night summary: {e}")

    # --- Always process and save hourly group SHAP data --- 
    # if args.group_shap_data:
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
    # Arguments to control which data files are generated - REMOVED as --all is always used
    # parser.add_argument(
    #     "--day-night-summary",
    #     action="store_true",
    #     default=False,
    #     help="Generate the day/night summary CSV.",
    # )
    # parser.add_argument(
    #     "--group-shap-data",
    #     action="store_true",
    #     default=False,
    #     help="Generate the hourly group SHAP and feature data CSVs by climate zone.",
    # )
    # parser.add_argument(
    #     "--all",
    #     action="store_true",
    #     default=False,
    #     help="Run all data generation steps (day/night summary, hourly group data).",
    # )
    return parser.parse_args()

if __name__ == "__main__":
    main()
