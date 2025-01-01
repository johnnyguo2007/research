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
from mlflow_tools import (
    generate_summary_and_kg_plots,
    create_day_night_summary,
    plot_shap_stacked_bar,
    plot_feature_group_stacked_bar,
    get_feature_groups,
    get_latex_label,
    generate_group_shap_plots_by_climate_zone,
)
from mlflow_tools.group_data import calculate_group_shap_values, GroupData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_feature_values(shap_values_feather_path: str) -> pd.DataFrame:
    """
    Loads feature values from the shap_values_with_additional_columns.feather file.

    Args:
        shap_values_feather_path (str): Path to the shap_values_with_additional_columns.feather file.

    Returns:
        pd.DataFrame: DataFrame containing feature values, aligned with SHAP values.
    """
    # Load the shap values with additional columns
    shap_df = pd.read_feather(shap_values_feather_path)

    # Identify feature columns: columns not in exclude list and not ending with '_shap'
    exclude_cols = [
        "global_event_ID",
        "lon",
        "lat",
        "time",
        "KGClass",
        "KGMajorClass",
        "local_hour",
        "UHI_diff",
        "Estimation_Error",
    ]
    feature_cols = [
        col
        for col in shap_df.columns
        if col not in exclude_cols and not col.endswith("_shap")
    ]

    # Extract the required columns
    feature_values_df = shap_df[
        [
            "global_event_ID",
            "lon",
            "lat",
            "time",
            "KGClass",
            "KGMajorClass",
            "local_hour",
        ]
        + feature_cols
    ]

    # Melt the feature values dataframe to long format
    feature_values_melted = feature_values_df.melt(
        id_vars=[
            "global_event_ID",
            "lon",
            "lat",
            "time",
            "KGClass",
            "KGMajorClass",
            "local_hour",
        ],
        value_vars=feature_cols,
        var_name="Feature",
        value_name="FeatureValue",
    )

    return feature_values_melted

def get_top_features(df: pd.DataFrame, top_n: int) -> List[str]:
    """
    Gets the list of top features based on total contribution.

    Args:
        df (pd.DataFrame): The DataFrame containing features.
        top_n (int): The number of top features to select.

    Returns:
        list: List of top feature names.
    """
    feature_totals = df.groupby("Feature")["Value"].sum()
    top_features_list = (
        feature_totals.sort_values(ascending=False).head(top_n).index.tolist()
    )
    return top_features_list



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

def prepare_shap_data(
    shap_df: pd.DataFrame,
    feature_values_df: pd.DataFrame,
    kg_class: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares SHAP data for plotting by filtering and organizing values.

    Args:
        shap_df (pd.DataFrame): DataFrame containing SHAP values
        feature_values_df (pd.DataFrame): DataFrame containing feature values
        kg_class (str, optional): Climate zone to filter by

    Returns:
        tuple: (shap_plot_df, feature_values_plot_df)
    """
    # Filter by climate zone if specified
    if kg_class and kg_class != "global":
        mask = feature_values_df["KGMajorClass"] == kg_class
        shap_df = shap_df[mask]
        feature_values_df = feature_values_df[mask]

    # Get SHAP columns and corresponding feature columns
    shap_cols = [col for col in shap_df.columns if col.endswith("_shap")]
    feature_cols = [col.replace("_shap", "") for col in shap_cols]

    return shap_df[shap_cols], feature_values_df[feature_cols]

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

def plot_stacked_bar(
    df: pd.DataFrame,
    group_by_column: str,
    output_path: str,
    title: str,
    color_mapping: Optional[Dict[str, str]] = None,
    base_value: float = 0,
    show_total_line: bool = True,
) -> None:
    """
    Plots a stacked bar chart with optional total line.

    Args:
        df (pd.DataFrame): DataFrame to plot.
        title (str): Title of the plot.
        output_path (str): Path to save the plot.
        color_mapping (dict, optional): Mapping of features to colors.
        base_value (float, optional): Base value for the plot.
        show_total_line (bool, optional): Whether to show the total line.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = (
        [color_mapping.get(feature, "#333333") for feature in df.columns]
        if color_mapping
        else None
    )
    df.plot(kind="bar", stacked=True, color=colors, ax=ax, bottom=base_value)

    if show_total_line:
        total_values = df.sum(axis=1) + base_value
        total_values.plot(
            kind="line", color="black", marker="o", linewidth=2, ax=ax, label="Total"
        )

    ax.set_title(title)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Value")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


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
    shap_values_feather_path = os.path.join(
        artifact_uri, "shap_values_with_additional_columns.feather"
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
        generate_summary_and_kg_plots(
            obj_group_data,
            output_dir,
            plot_type="feature",
        )
        generate_summary_and_kg_plots(
            obj_group_data,
            output_dir,
            plot_type="group",
        )

    # Generate day/night summary if requested
    if args.day_night_summary:
        summary_df = create_day_night_summary(obj_group_data.shap_detail_df, all_df, output_dir)
        logging.info("\nFeature Group Day/Night Summary:")
        logging.info("\n" + str(summary_df))

    raw_df = all_df
    # base_value is different for each hour, so we need to get the base_value for each hour 
    base_values: pd.Series = raw_df.groupby("local_hour")["base_value"].first()

    # group all_df by local_hour and KGMajorClass, exclude  "global_event_ID",  "lon",  "lat",  "time", "KGClass",
    exclude_cols = ["global_event_ID", "lon", "lat", "time", "KGClass"]
    #drop the exclude_cols then group by local_hour and KGMajorClass    
    all_df_by_hour_kg = raw_df.drop(columns=exclude_cols).groupby(["local_hour", "KGMajorClass"]).mean().reset_index()
    
    # Calculate group values for the grouped data
    obj_group_data = calculate_group_shap_values(all_df_by_hour_kg, feature_to_group_mapping)

    # # Load and prepare feature values for plotting
    # feature_values_melted = load_feature_values(shap_values_feather_path)

    # Generate plots for each climate zone
    kg_classes = ["global"] + all_df_by_hour_kg["KGMajorClass"].unique().tolist()
    if args.group_shap_plots:
        for kg_class in kg_classes:
            # Prepare data for current climate zone
            feature_group_data = obj_group_data.shap_group_hourly_mean_df(kg_class)

            # Generate stacked bar plot
            plot_title = (
                "Global Feature Group Contribution by Hour"
                if kg_class == "global"
                else f"Feature Group Contribution by Hour for {kg_class}"
            )
            output_path = os.path.join(
                output_dir, f"feature_group_contribution_by_hour_{kg_class}.png"
            )

            plot_feature_group_stacked_bar(
                feature_group_data, "local_hour", output_path, plot_title, base_values
            )

            # Generate SHAP and feature value plots
            generate_group_shap_plots_by_climate_zone(
                obj_group_data=obj_group_data,
                kg_classes=[kg_class],
                output_dir=output_dir,
                base_values=base_values,
                show_total_feature_line=not args.hide_total_feature_line,
            )

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
        required=True,
        help="Name of the MLflow experiment to process.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=None,
        help="Number of top features to plot. If None, plot all features.",
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
        help="Run all analyses (summary plots, day/night summary, and group SHAP plots).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()