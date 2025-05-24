import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import logging
import mlflow
import seaborn as sns
import shap
from typing import List, Dict, Tuple, Optional
sys.path.append('/home/jguo/research/hw_global/ultimate/')

from mlflow_tools import (
    generate_summary_and_kg_plots,
    create_day_night_summary,
    plot_shap_stacked_bar,
    plot_feature_group_stacked_bar,
    get_feature_groups,
    get_latex_label,
    get_label_with_unit,
    generate_group_shap_plots_by_climate_zone,
)
from mlflow_tools.group_data import calculate_group_shap_values, GroupData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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

def main():

    ''' 
    Sample command:
    python paper_figure_feature_contribution_by_hour.py --experiment-name Combined_Final_noFSA_Hourly_HW98_Hour 2>&1 | tee rlog
    '''

    """Main function to process SHAP values and generate plots."""
    logging.info("Starting feature contribution analysis...")

    # Parse arguments
    args = parse_arguments()
    logging.info(f"Parsed command line arguments: {vars(args)}")

    # Set flags based on --all argument if provided
    if args.all:
        args.summary_plots = True
        args.day_night_summary = True
        args.group_shap_plots = True
        args.plot_fsh_interaction_global = True

    # Get experiment and run
    experiment_id, run_id = get_experiment_and_run(args.experiment_name)
    if experiment_id is None or run_id is None:
        return

    # Setup paths
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    artifact_uri = artifact_uri.replace(
        "mlflow-artifacts:",
        "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts",
    )
    feather_file_name = args.feather_file_name
    logging.info(f"Using feather file: {feather_file_name}")
    shap_values_feather_path = os.path.join(
        artifact_uri, feather_file_name
    )
    output_dir = os.path.join(artifact_uri, "24_hourly_plot")
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    all_df = pd.read_feather(shap_values_feather_path)
    base_value = all_df["base_value"].iloc[0] if "base_value" in all_df.columns else 0

    # Get feature names and groups
    shap_cols, feature_cols = extract_shap_and_feature_columns(all_df)
    feature_to_group_mapping = get_feature_groups(feature_cols)
    shap_df = all_df[shap_cols]
    feature_values_df = all_df[feature_cols]

    # Calculate group values
    obj_group_data = calculate_group_shap_values(all_df, feature_to_group_mapping)
    
    logging.info(f"all_df columns: {all_df.columns.tolist()}, shape: {all_df.shape}")
    logging.info(f"shap_cols: {shap_cols}, shape: {shap_df.shape}")
    logging.info(f"feature_cols: {feature_cols}, shape: {feature_values_df.shape}")
    # logging.info(f"group_data df columns: {obj_group_data.df.columns.tolist()}, shape: {obj_group_data.df.shape}")
    logging.info(f"group_names: {obj_group_data.group_names}")

    # Generate summary plots if requested
    if args.summary_plots:
        logging.info("Processing summary plots with day/night separation...")
        day_hours = set(range(7, 19))  # 7 AM to 6 PM (18:00) inclusive

        if 'local_hour' not in all_df.columns:
            logging.error("'local_hour' column not found in all_df. Cannot separate day/night for summary plots.")
        else:
            # Filter all_df for daytime
            day_df = all_df[all_df['local_hour'].isin(day_hours)].copy()
            # Filter all_df for nighttime
            night_df = all_df[~all_df['local_hour'].isin(day_hours)].copy()

            # --- Daytime Plots ---
            if not day_df.empty:
                logging.info("Generating daytime summary plots...")
                day_obj_group_data = calculate_group_shap_values(day_df, feature_to_group_mapping)
                
                day_output_sub_dir_name = "daytime_summary_plots"
                day_specific_output_dir = os.path.join(output_dir, day_output_sub_dir_name)
                os.makedirs(day_specific_output_dir, exist_ok=True)
                
                logging.info(f"Daytime data shape for summary: {day_df.shape}")
                if hasattr(day_obj_group_data, 'group_names'):
                    logging.info(f"Daytime GroupData groups for summary: {day_obj_group_data.group_names}")

                generate_summary_and_kg_plots(
                    day_obj_group_data,
                    day_specific_output_dir, 
                    plot_type="feature",
                )
                generate_summary_and_kg_plots(
                    day_obj_group_data,
                    day_specific_output_dir,
                    plot_type="group",
                )
            else:
                logging.warning("No daytime data found (hours 7-18). Skipping daytime summary plots.")

            # --- Nighttime Plots ---
            if not night_df.empty:
                logging.info("Generating nighttime summary plots...")
                night_obj_group_data = calculate_group_shap_values(night_df, feature_to_group_mapping)

                night_output_sub_dir_name = "nighttime_summary_plots"
                night_specific_output_dir = os.path.join(output_dir, night_output_sub_dir_name)
                os.makedirs(night_specific_output_dir, exist_ok=True)

                logging.info(f"Nighttime data shape for summary: {night_df.shape}")
                if hasattr(night_obj_group_data, 'group_names'):
                    logging.info(f"Nighttime GroupData groups for summary: {night_obj_group_data.group_names}")

                generate_summary_and_kg_plots(
                    night_obj_group_data,
                    night_specific_output_dir,
                    plot_type="feature",
                )
                generate_summary_and_kg_plots(
                    night_obj_group_data,
                    night_specific_output_dir,
                    plot_type="group",
                )
            else:
                logging.warning("No nighttime data found (hours outside 7-18). Skipping nighttime summary plots.")

    # Generate day/night summary if requested
    if args.day_night_summary:
        summary_df = create_day_night_summary(obj_group_data.shap_detail_df, all_df, output_dir)
        logging.info("\nFeature Group Day/Night Summary:")
        logging.info("\n" + str(summary_df))

    raw_df = all_df
    # Calculate base values globally (averaged over all zones for each hour)
    global_base_values: pd.Series = raw_df.groupby("local_hour")["base_value"].first()
    # Calculate base values grouped by hour and climate zone
    base_values_by_hour_kg: pd.DataFrame = raw_df.groupby(["local_hour", "KGMajorClass"])["base_value"].first().reset_index()

    # group all_df by local_hour and KGMajorClass, exclude  "global_event_ID",  "lon",  "lat",  "time", "KGClass",
    exclude_cols = ["global_event_ID", "lon", "lat", "time", "KGClass"]
    #drop the exclude_cols then group by local_hour and KGMajorClass
    all_df_by_hour_kg = raw_df.drop(columns=exclude_cols).groupby(["local_hour", "KGMajorClass"]).mean().reset_index()

    # Calculate group values for the grouped data
    obj_group_data = calculate_group_shap_values(all_df_by_hour_kg, feature_to_group_mapping)

    # Generate plots for each climate zone
    kg_classes = ["global"] + sorted(all_df_by_hour_kg["KGMajorClass"].unique().tolist()) # Sort for consistent order
    if args.group_shap_plots:
        for kg_class in kg_classes:
            # Prepare data for current climate zone
            feature_group_data = obj_group_data.shap_group_hourly_mean_df(kg_class)

            # Determine the appropriate base values for the current plot
            if kg_class == "global":
                current_base_values = global_base_values
            else:
                # Filter base values for the specific climate zone
                kg_base_values_df = base_values_by_hour_kg[base_values_by_hour_kg["KGMajorClass"] == kg_class]
                if kg_base_values_df.empty:
                    logging.warning(f"No base values found for kg_class {kg_class}. Using global base values.")
                    current_base_values = global_base_values
                else:
                    # Convert to a Series indexed by local_hour
                    current_base_values = kg_base_values_df.set_index("local_hour")["base_value"]


            # Generate stacked bar plot
            plot_title = (
                "Global Feature Group Contribution by Hour"
                if kg_class == "global"
                else f"Feature Group Contribution by Hour for {get_latex_label(kg_class)}" # Use latex label
            )
            output_path = os.path.join(
                output_dir, f"feature_group_contribution_by_hour_{kg_class}.png"
            )

            plot_feature_group_stacked_bar(
                feature_group_data, "local_hour", output_path, plot_title, current_base_values # Use current_base_values
            )

            # Generate SHAP and feature value plots
            generate_group_shap_plots_by_climate_zone(
                obj_group_data=obj_group_data,
                kg_classes=[kg_class],
                output_dir=output_dir,
                base_values=current_base_values, # Use current_base_values
                show_total_feature_line=not args.hide_total_feature_line,
            )

    # Generate FSH interaction plots if requested
    if args.plot_fsh_interaction_global:
        logging.info("Generating FSH SHAP interaction plots for global (day/night)...")
        day_hours = set(range(7, 19))  # 7 AM to 6 PM (18:00) inclusive

        # Define the specific FSH column names
        fsh_feature_col = "hw_nohw_diff_FSH"
        fsh_shap_col = "hw_nohw_diff_FSH_shap"

        if 'local_hour' not in all_df.columns:
            logging.error("'local_hour' column not found. Cannot create day/night FSH interaction plots.")
        elif fsh_feature_col not in all_df.columns or fsh_shap_col not in all_df.columns:
            logging.error(f"'{fsh_feature_col}' or '{fsh_shap_col}' column not found. Cannot create FSH interaction plots.")
        else:
            # Prepare dataframes for day and night
            day_df_interaction = all_df[all_df['local_hour'].isin(day_hours)].copy()
            night_df_interaction = all_df[~all_df['local_hour'].isin(day_hours)].copy()

            # Get the list of other features to interact with
            other_feature_cols_for_interaction = [col for col in feature_cols if col != fsh_feature_col]

            # Plot for daytime
            if not day_df_interaction.empty:
                logging.info(f"Generating daytime FSH interaction plots against all other features ({len(other_feature_cols_for_interaction)} plots).")
                shap_values_day = day_df_interaction[shap_cols].values
                features_day = day_df_interaction[feature_cols]
                for interaction_feature in other_feature_cols_for_interaction:
                    if interaction_feature not in features_day.columns:
                        logging.warning(f"Interaction feature '{interaction_feature}' not found in daytime data. Skipping.")
                        continue
                    plt.figure(figsize=(10, 8))
                    try:
                        shap.dependence_plot(
                            fsh_feature_col,  # Feature whose SHAP values are on y-axis
                            shap_values_day,
                            features_day,
                            interaction_index=interaction_feature, # Feature to color by
                            show=False
                        )
                        fsh_latex = get_latex_label(fsh_feature_col)
                        interaction_latex = get_latex_label(interaction_feature)

                        # Use get_label_with_unit for x-axis
                        x_axis_label = get_label_with_unit(fsh_feature_col)

                        # Define SHAP values unit (assuming Kelvin for model output units)
                        # This should be configured if model output unit is different
                        shap_values_unit_latex = "K"
                        y_axis_label = f"SHAP value for {fsh_latex} ({shap_values_unit_latex})"

                        plt.xlabel(x_axis_label)
                        plt.ylabel(y_axis_label)

                        plt.title(f"{fsh_latex} SHAP Interaction with {interaction_latex} (Global - Daytime)")
                        day_interaction_plot_path = os.path.join(output_dir, f"fsh_interaction_with_{interaction_feature.replace('/', '_')}_global_daytime.png") # Sanitize filename
                        plt.savefig(day_interaction_plot_path)
                        plt.close()
                        logging.info(f"Daytime FSH interaction plot with {interaction_latex} saved to: {day_interaction_plot_path}")
                    except Exception as e:
                        logging.error(f"Error generating daytime FSH interaction plot with {interaction_feature}: {e}")
                        plt.close() # Ensure plot is closed on error
            else:
                logging.warning("No daytime data for FSH interaction plot.")

            # Plot for nighttime
            if not night_df_interaction.empty:
                logging.info(f"Generating nighttime FSH interaction plots against all other features ({len(other_feature_cols_for_interaction)} plots).")
                shap_values_night = night_df_interaction[shap_cols].values
                features_night = night_df_interaction[feature_cols]
                for interaction_feature in other_feature_cols_for_interaction:
                    if interaction_feature not in features_night.columns:
                        logging.warning(f"Interaction feature '{interaction_feature}' not found in nighttime data. Skipping.")
                        continue
                    plt.figure(figsize=(10, 8))
                    try:
                        shap.dependence_plot(
                            fsh_feature_col,  # Feature whose SHAP values are on y-axis
                            shap_values_night,
                            features_night,
                            interaction_index=interaction_feature, # Feature to color by
                            show=False
                        )
                        fsh_latex = get_latex_label(fsh_feature_col)
                        interaction_latex = get_latex_label(interaction_feature)

                        # Use get_label_with_unit for x-axis
                        x_axis_label = get_label_with_unit(fsh_feature_col)

                        # Define SHAP values unit (assuming Kelvin for model output units)
                        # This should be configured if model output unit is different
                        shap_values_unit_latex = "K"
                        y_axis_label = f"SHAP value for {fsh_latex} ({shap_values_unit_latex})"

                        plt.xlabel(x_axis_label)
                        plt.ylabel(y_axis_label)

                        plt.title(f"{fsh_latex} SHAP Interaction with {interaction_latex} (Global - Nighttime)")
                        night_interaction_plot_path = os.path.join(output_dir, f"fsh_interaction_with_{interaction_feature.replace('/', '_')}_global_nighttime.png") # Sanitize filename
                        plt.savefig(night_interaction_plot_path)
                        plt.close()
                        logging.info(f"Nighttime FSH interaction plot with {interaction_latex} saved to: {night_interaction_plot_path}")
                    except Exception as e:
                        logging.error(f"Error generating nighttime FSH interaction plot with {interaction_feature}: {e}")
                        plt.close() # Ensure plot is closed on error
            else:
                logging.warning("No nighttime data for FSH interaction plot.")

    logging.info("Analysis completed successfully.")
    logging.info(f"All outputs have been saved to: {output_dir}")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Report and plot SHAP value contributions by feature group and hour."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Combined_Final3_NO_LE_Hourly_HW98_Hour",
        help="Name of the MLflow experiment to process.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=None,
        help="Number of top features to plot. If None, plot all features.",
    )
    parser.add_argument(
        "--feather-file-name",
        type=str,
        default="shap_values_with_additional_columns.feather",
        help="Name of the input feather file containing SHAP values.",
    )
    parser.add_argument(
        "--hide-total-feature-line",
        action="store_true",
        help="Hide the total feature value line in feature value plots.",
    )
    parser.add_argument(
        "--day-night-summary",
        action="store_true",
        default=False,
        help="Generate the day/night summary.",
    )
    parser.add_argument(
        "--summary-plots",
        action="store_true",
        default=False,
        help="Generate the summary plots.",
    )
    parser.add_argument(
        "--group-shap-plots",
        action="store_true",
        default=False,
        help="Generate the group SHAP plots by climate zone.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all analyses (summary plots, day/night summary, group SHAP plots, and FSH interaction plots).",
    )
    parser.add_argument(
        "--plot_fsh_interaction_global",
        action="store_true",
        default=False,
        help="Generate SHAP interaction plot for FSH (global, day/night).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()