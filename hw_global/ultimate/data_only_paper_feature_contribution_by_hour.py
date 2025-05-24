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

def _calculate_importance_df(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Calculate feature importance from SHAP values, following the same logic as plot_summary.py get_shap_feature_importance.
    
    Args:
        shap_values: NumPy array of SHAP values (samples x features)
        feature_names: List of feature names
        
    Returns:
        DataFrame with columns: Feature, Importance, Percentage (sorted by Importance descending)
    """
    # Debug information
    logging.info(f"SHAP values shape: {shap_values.shape}")
    logging.info(f"Number of feature names: {len(feature_names)}")
    logging.info(f"Feature names: {feature_names[:5]}...")  # Show first 5 feature names
    
    # Ensure we have the right number of features
    if shap_values.shape[1] != len(feature_names):
        logging.error(f"Mismatch: SHAP values has {shap_values.shape[1]} features but feature_names has {len(feature_names)} features")
        # Use the minimum to avoid index errors
        min_features = min(shap_values.shape[1], len(feature_names))
        shap_values = shap_values[:, :min_features]
        feature_names = feature_names[:min_features]
        logging.warning(f"Truncated to {min_features} features")
    
    shap_feature_importance = np.abs(shap_values).mean(axis=0)
    total_importance = np.sum(shap_feature_importance)
    
    # Ensure all arrays have the same length
    assert len(feature_names) == len(shap_feature_importance), f"Length mismatch: {len(feature_names)} vs {len(shap_feature_importance)}"
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_feature_importance,
        'Percentage': (shap_feature_importance / total_importance) * 100
    })
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    return importance_df

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

    # Get all columns, sort alphabetically, ensuring 'Total', 'UHI_diff', 'y_pred', and 'Estimation_Error' are last
    cols = output_df.columns.tolist()
    has_total = "Total" in cols
    has_uhi = "UHI_diff" in cols
    has_ypred = "y_pred" in cols
    has_error = "Estimation_Error" in cols # ADDED: Check for Estimation_Error

    special_cols = [] # List to hold the special columns in desired order
    if has_total:
        cols.remove("Total")
        special_cols.append("Total")
    if has_uhi:
        cols.remove("UHI_diff")
        special_cols.append("UHI_diff")
    if has_ypred:
        cols.remove("y_pred")
        special_cols.append("y_pred")
    if has_error:
        cols.remove("Estimation_Error")
        special_cols.append("Estimation_Error") # ADDED: Add Estimation_Error to special cols

    cols.sort() # Sort the remaining columns alphabetically

    # Append the special columns at the end
    cols.extend(special_cols)

    # Reindex the DataFrame with the desired column order
    output_df = output_df[cols]

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
    base_values: pd.Series,
    all_df: pd.DataFrame, # Added: Original full dataframe for global UHI calc
    all_df_by_hour_kg: pd.DataFrame # Added: Hourly aggregated dataframe for KG UHI calc
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
        # --- Add UHI_diff / y_pred / Estimation_Error columns --- # MODIFIED
        uhi_diff_hourly = None
        y_pred_hourly = None
        estimation_error_hourly = None # ADDED: Initialize estimation_error_hourly

        if kg_class == "global":
            # Calculate global hourly means from the original df
            if 'UHI_diff' in all_df.columns and 'local_hour' in all_df.columns:
                 uhi_diff_hourly = all_df.groupby('local_hour')['UHI_diff'].mean()
            else:
                 logging.warning("Could not find 'UHI_diff' or 'local_hour' in all_df for global UHI_diff calculation.")
            if 'y_pred' in all_df.columns and 'local_hour' in all_df.columns:
                 y_pred_hourly = all_df.groupby('local_hour')['y_pred'].mean()
            else:
                 logging.warning("Could not find 'y_pred' or 'local_hour' in all_df for global y_pred calculation.")
            # ADDED: Calculate global Estimation_Error
            if 'Estimation_Error' in all_df.columns and 'local_hour' in all_df.columns:
                 estimation_error_hourly = all_df.groupby('local_hour')['Estimation_Error'].mean()
            else:
                 logging.warning("Could not find 'Estimation_Error' or 'local_hour' in all_df for global Estimation_Error calculation.")
        else:
            # Get KG-specific hourly means from the pre-aggregated df
            required_cols = ['UHI_diff', 'y_pred', 'Estimation_Error', 'KGMajorClass', 'local_hour'] # ADDED Estimation_Error
            if all(col in all_df_by_hour_kg.columns for col in required_cols):
                kg_hourly_df = all_df_by_hour_kg[all_df_by_hour_kg['KGMajorClass'] == kg_class]
                if not kg_hourly_df.empty:
                    uhi_diff_hourly = kg_hourly_df.set_index('local_hour')['UHI_diff']
                    y_pred_hourly = kg_hourly_df.set_index('local_hour')['y_pred']
                    estimation_error_hourly = kg_hourly_df.set_index('local_hour')['Estimation_Error'] # ADDED: Get Estimation_Error
                else:
                    logging.warning(f"No data found for KGMajorClass '{kg_class}' in all_df_by_hour_kg.")
            else:
                missing_cols = [col for col in required_cols if col not in all_df_by_hour_kg.columns]
                logging.warning(f"Could not find required columns ({', '.join(missing_cols)}) in all_df_by_hour_kg for {kg_class}.")

        # Join UHI_diff if successfully calculated
        if uhi_diff_hourly is not None and not uhi_diff_hourly.empty:
            try:
                shap_grouped_features_df.index = shap_grouped_features_df.index.astype(uhi_diff_hourly.index.dtype)
            except Exception as e:
                logging.warning(f"Could not align index types for joining UHI_diff for {kg_class}: {e}")
            shap_grouped_features_df = shap_grouped_features_df.join(uhi_diff_hourly.rename('UHI_diff'))
            logging.info(f"Added 'UHI_diff' column for {kg_class}.")
        else:
            logging.warning(f"Could not calculate or join 'UHI_diff' for {kg_class}.")

        # Join y_pred if successfully calculated
        if y_pred_hourly is not None and not y_pred_hourly.empty:
            try:
                if 'UHI_diff' not in shap_grouped_features_df.columns: # Only align if UHI_diff didn't
                     shap_grouped_features_df.index = shap_grouped_features_df.index.astype(y_pred_hourly.index.dtype)
            except Exception as e:
                logging.warning(f"Could not align index types for joining y_pred for {kg_class}: {e}")
            shap_grouped_features_df = shap_grouped_features_df.join(y_pred_hourly.rename('y_pred'))
            logging.info(f"Added 'y_pred' column for {kg_class}.")
        else:
            logging.warning(f"Could not calculate or join 'y_pred' for {kg_class}.")

        # ADDED: Join Estimation_Error if successfully calculated
        if estimation_error_hourly is not None and not estimation_error_hourly.empty:
            try:
                # Only align if UHI_diff and y_pred didn't
                if 'UHI_diff' not in shap_grouped_features_df.columns and 'y_pred' not in shap_grouped_features_df.columns:
                     shap_grouped_features_df.index = shap_grouped_features_df.index.astype(estimation_error_hourly.index.dtype)
            except Exception as e:
                 logging.warning(f"Could not align index types for joining Estimation_Error for {kg_class}: {e}")
            shap_grouped_features_df = shap_grouped_features_df.join(estimation_error_hourly.rename('Estimation_Error'))
            logging.info(f"Added 'Estimation_Error' column for {kg_class}.")
        else:
            logging.warning(f"Could not calculate or join 'Estimation_Error' for {kg_class}.")
        # --- End Add columns ---

        # Define base path for this file using the DISPLAY name
        output_path_base_grouped = os.path.join(kg_class_dir, f"{kg_class_display_name}") # e.g., .../Continental/Continental
        
        # Calculate the Total column (Sum of group SHAP + Base Value for each hour)
        # Ensure base_values index aligns with shap_grouped_features_df index (local_hour)
        if not base_values.index.equals(shap_grouped_features_df.index):
             logging.warning(f"Index mismatch between base_values and grouped SHAP for {kg_class}. Cannot calculate Total reliably. Saving without Total.")
             total_values_with_base = None
        else:
             # Ensure UHI_diff, y_pred, and Estimation_Error are excluded from the sum if they exist before adding base_values
             cols_to_exclude_from_sum = ['UHI_diff', 'y_pred', 'Estimation_Error'] # ADDED Estimation_Error
             total_values_with_base = shap_grouped_features_df.drop(columns=cols_to_exclude_from_sum, errors='ignore').sum(axis=1) + base_values

        # Save grouped SHAP values using a descriptive data_type and include the total
        _save_data_csv(shap_grouped_features_df, output_path_base_grouped, "group_shap_contribution", total_values=total_values_with_base) # Saves as ..._group_shap_contribution_data.csv


def save_feature_importance_data(
    all_df: pd.DataFrame,
    kg_class: str,
    day_night: str,  # "day" or "night"
    output_dir: str,
    feature_to_group_mapping: Dict[str, str]
) -> None:
    """
    Calculate and save feature importance data (mean SHAP values) for a specific climate zone and time period.
    
    Args:
        all_df: DataFrame containing SHAP values for the specific KG class and time period
        kg_class: Climate zone class (e.g., 'Arid', 'Cold', 'global')
        day_night: Time period - either "day" or "night"
        output_dir: Base output directory
        feature_to_group_mapping: Mapping of features to groups
    """
    if all_df.empty:
        logging.warning(f"Empty DataFrame for {kg_class} {day_night}. Skipping feature importance data.")
        return
    
    # Get display name for paths
    kg_class_display_name = replace_cold_with_continental(kg_class)
    
    # Create subdirectory structure matching the figure script
    time_dir = f"{day_night}time_feature_summary_plots"
    kg_dir = os.path.join(output_dir, time_dir, kg_class_display_name)
    os.makedirs(kg_dir, exist_ok=True)
    
    # Create GroupData object to match main script's data processing exactly
    obj_group_data = calculate_group_shap_values(all_df, feature_to_group_mapping)
    
    # Get data the same way as the main script for individual features
    if kg_class == "global":
        shap_df = obj_group_data.shap_detail_df.copy()
        feature_values_df = obj_group_data.feature_detail_df
        # feature_names = obj_group_data.feature_cols_names # Not directly used for importance calc
    else:
        # Filter for specific kg_class the same way as main script
        kg_mask = obj_group_data.df["KGMajorClass"] == kg_class
        shap_df = obj_group_data.shap_detail_df[kg_mask].copy()
        feature_values_df = obj_group_data.feature_detail_df[kg_mask]
        # feature_names = obj_group_data.feature_cols_names # Not directly used for importance calc
    
    # Debug information
    logging.info(f"SHAP DataFrame shape for importance calc ({kg_class} - {day_night}): {shap_df.shape}")
    
    # Use the actual columns from the SHAP DataFrame instead of feature_names
    actual_feature_names = shap_df.columns.tolist()
    logging.info(f"Using actual SHAP DataFrame columns as feature names: {len(actual_feature_names)} features for {kg_class} - {day_night}")
    
    # Calculate feature importance using the processed data
    shap_values = shap_df.values
    # _calculate_importance_df returns a DataFrame with 'Feature', 'Importance', 'Percentage'
    individual_importance_df = _calculate_importance_df(shap_values, actual_feature_names) 
    
    # --- Save individual SHAP importance data in the new format ---
    feature_importance_base_path_for_individual = os.path.join(kg_dir, f"feature_importance_{kg_class_display_name}_{day_night}")
    # individual_importance_df directly contains 'Feature', 'Importance', 'Percentage'
    csv_path_individual_shap_importance = f"{feature_importance_base_path_for_individual}_individual_shap_importance.csv" # Changed suffix for clarity
    individual_importance_df[['Feature', 'Importance', 'Percentage']].to_csv(csv_path_individual_shap_importance, index=False)
    logging.info(f"Saved individual SHAP importance data to {csv_path_individual_shap_importance}")

    # --- Prepare and save individual feature values (this part remains mostly as is) ---
    ordered_feature_cols = individual_importance_df['Feature'].tolist()
    
    available_feature_cols = [col for col in ordered_feature_cols if col in feature_values_df.columns]
    if len(available_feature_cols) != len(ordered_feature_cols):
        logging.warning(f"Some features missing in feature_values_df for {kg_class} {day_night}. Available: {len(available_feature_cols)}, Expected from importance: {len(ordered_feature_cols)}")
        logging.warning(f"Missing features: {set(ordered_feature_cols) - set(available_feature_cols)}") # Corrected missing set
        # Use only available_feature_cols for feature_means if there's a mismatch
        ordered_feature_cols_for_means = available_feature_cols 
    else:
        ordered_feature_cols_for_means = ordered_feature_cols

    if ordered_feature_cols_for_means and not feature_values_df.empty: # Check if feature_values_df is not empty
        feature_means = feature_values_df[ordered_feature_cols_for_means].mean()
    else:
        feature_means = pd.Series(dtype=float) # Empty series if no features or empty df
        logging.warning(f"No feature means to calculate for {kg_class} {day_night} as ordered_feature_cols_for_means is empty or feature_values_df is empty.")

    feature_values_df_output = pd.DataFrame({
        feature: feature_means[feature] for feature in ordered_feature_cols_for_means if feature in feature_means.index
    }, index=['mean_value'])
    
    metadata_cols = ['base_value', 'UHI_diff', 'y_pred', 'Estimation_Error']
    if not all_df.empty: # Ensure all_df is not empty before accessing columns
        for col in metadata_cols:
            if col in all_df.columns:
                mean_val = all_df[col].mean()
                feature_values_df_output[col] = mean_val
    
    # Save individual feature values using the existing _save_data_csv function
    # feature_importance_base_path_for_individual defined above
    _save_data_csv(feature_values_df_output, feature_importance_base_path_for_individual, "individual_feature_values")
    
    # Calculate group-level importance data using the same method
    if kg_class == "global":
        group_shap_df = obj_group_data.shap_group_detail_df.copy()
        # group_feature_names = obj_group_data.group_names # Not directly used
    else:
        kg_mask = obj_group_data.df["KGMajorClass"] == kg_class
        group_shap_df = obj_group_data.shap_group_detail_df[kg_mask].copy()
        # group_feature_names = obj_group_data.group_names # Not directly used
    
    actual_group_names = group_shap_df.columns.tolist()
    logging.info(f"Group SHAP DataFrame shape for importance calc ({kg_class} - {day_night}): {group_shap_df.shape}")
    logging.info(f"Using actual group SHAP DataFrame columns: {actual_group_names} for {kg_class} - {day_night}")
    
    group_shap_values = group_shap_df.values
    # _calculate_importance_df returns 'Feature' (which are group names here), 'Importance', 'Percentage'
    group_importance_df = _calculate_importance_df(group_shap_values, actual_group_names) 
    
    # --- Save group SHAP importance data in the new format ---
    group_importance_to_save = group_importance_df.rename(
        columns={'Feature': 'Group'}
    )[['Group', 'Importance', 'Percentage']].copy()
    
    time_dir_group = f"{day_night}time_group_summary_plots"
    kg_dir_group = os.path.join(output_dir, time_dir_group, kg_class_display_name)
    os.makedirs(kg_dir_group, exist_ok=True)
    
    group_importance_base_path = os.path.join(kg_dir_group, f"group_importance_{kg_class_display_name}_{day_night}")
    csv_path_group_shap_importance = f"{group_importance_base_path}_group_shap_importance.csv" # Changed suffix for clarity
    group_importance_to_save.to_csv(csv_path_group_shap_importance, index=False)
    logging.info(f"Saved group SHAP importance data to {csv_path_group_shap_importance}")
    
    logging.info(f"Saved feature and group importance data for {kg_class_display_name} ({day_night})")


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

    feather_file_name = args.feather_file_name
    logging.info(f"Using feather file: {feather_file_name}")
    shap_values_feather_path = os.path.join(
        artifact_uri, feather_file_name
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
        # Add y_pred column as sum of UHI_diff and Estimation_Error
        if "UHI_diff" in all_df.columns and "Estimation_Error" in all_df.columns:
            all_df["y_pred"] = all_df["UHI_diff"] + all_df["Estimation_Error"]
            logging.info("Added 'y_pred' column to DataFrame.")
        else:
            logging.warning("Could not add 'y_pred' column: required columns 'UHI_diff' or 'Estimation_Error' missing.")
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

    # --- Generate feature importance data for each climate zone ---
    logging.info("Generating feature importance data for each climate zone...")
    
    # Define day/night hours
    day_hours = set(range(7, 19))  # 7 AM to 6 PM
    
    if 'local_hour' not in all_df.columns:
        logging.error("'local_hour' column not found. Cannot generate day/night feature importance data.")
    else:
        # Get unique climate zones
        kg_classes_for_importance = ["global"]
        if 'KGMajorClass' in all_df.columns:
            kg_classes_for_importance += sorted(all_df['KGMajorClass'].dropna().unique().tolist())
        
        for kg_class in kg_classes_for_importance:
            # Filter data for this climate zone
            if kg_class == "global":
                kg_df = all_df.copy()
            else:
                kg_df = all_df[all_df['KGMajorClass'] == kg_class].copy()
            
            if kg_df.empty:
                logging.warning(f"No data found for KGMajorClass '{kg_class}'. Skipping.")
                continue
            
            # Process day data
            day_df = kg_df[kg_df['local_hour'].isin(day_hours)]
            if not day_df.empty:
                save_feature_importance_data(
                    day_df, kg_class, "day", output_dir, feature_to_group_mapping
                )
            
            # Process night data
            night_df = kg_df[~kg_df['local_hour'].isin(day_hours)]
            if not night_df.empty:
                save_feature_importance_data(
                    night_df, kg_class, "night", output_dir, feature_to_group_mapping
                )

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
            base_values_hourly,
            all_df, # Pass original df
            all_df_by_hour_kg # Pass hourly aggregated df
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
        default="Combined_Final3_NO_LE_Hourly_HW98_Hour",
        help="Name of the MLflow experiment to process.",
    )
    parser.add_argument(
        "--feather-file-name",
        type=str,
        default="shap_values_with_additional_columns.feather",
        help="Name of the input feather file containing SHAP values.",
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
