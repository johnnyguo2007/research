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
    plot_shap_and_feature_values,
    plot_shap_and_feature_values_for_group
)

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




# Define the function with 'Value' instead of 'Importance'

def calculate_group_shap_values(
    shap_df: pd.DataFrame,
    feature_values_df: pd.DataFrame,
    feature_groups: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Calculate group-level SHAP values and feature values, returning DataFrames.

    Args:
        shap_df (pd.DataFrame): DataFrame containing SHAP values
        feature_values_df (pd.DataFrame): DataFrame containing feature values
        feature_groups (dict): Mapping from features to their groups

    Returns:
        tuple: (group_shap_df, group_feature_values_df, group_names)
    """
    # Get unique group names
    group_names: List[str] = list(set(feature_groups.values()))

    # Initialize empty DataFrames to store results
    group_shap_df = pd.DataFrame()
    group_feature_values_df = pd.DataFrame()

    for group in group_names:
        # Get features in this group
        group_features: List[str] = [f for f, g in feature_groups.items() if g == group]

        # Get corresponding SHAP columns
        group_shap_cols: List[str] = [f"{f}_shap" for f in group_features]

        # Sum SHAP values for features in the group, directly assign to new column
        group_shap_df[group] = shap_df[group_shap_cols].sum(axis=1)

        # Sum feature values for the group, directly assign to new column
        group_feature_values_df[group] = feature_values_df[group_features].sum(axis=1)

    return group_shap_df.reset_index(drop=True), group_feature_values_df.reset_index(drop=True), group_names


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


def prepare_feature_group_data(
    df_feature: pd.DataFrame, kg_class: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepares feature group data for plotting.

    Args:
        df_feature (pd.DataFrame): DataFrame containing feature data
        kg_class (str, optional): Climate zone to filter by

    Returns:
        pd.DataFrame: Processed feature group data
    """
    if kg_class and kg_class != "global":
        df_feature = df_feature[df_feature["KGMajorClass"] == kg_class]

    return df_feature


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
    """Main function to process SHAP values and generate plots."""
    logging.info("Starting feature contribution analysis...")

    # Parse arguments
    args = parse_arguments()
    logging.info(f"Parsed command line arguments: {vars(args)}")

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
    # the raw_df looks like this:
    # shap_cols, feature_val_cols, local_hour,  base_value,
        # "global_event_ID",
        # "lon",
        # "lat",
        # "time",
        # "KGClass",
        # "KGMajorClass",
        # "local_hour",
        # "UHI_diff",
        # "Estimation_Error",

    all_df = pd.read_feather(shap_values_feather_path)
    base_value = all_df["base_value"].iloc[0] if "base_value" in all_df.columns else 0

    # Get feature names and groups
    shap_cols, feature_cols = extract_shap_and_feature_columns(all_df)
    feature_groups = get_feature_groups(feature_cols)
    shap_df = all_df[shap_cols]
    feature_values_df = all_df[feature_cols]

    # the feature data looks like this:
    # shap_cols, feature_val_cols, local_hour, KGClass, KGMajorClass, global_event_ID, lon, lat, time, base_value
    # we report data based on local_hour and KGMajorClass, so we need to group by local_hour and KGMajorClass and then sum the shap_cols and feature_val_cols

    # the feature group data looks like this: please note there is not group_feature_val_cols, it is just feature_val_cols  
    # group_shap_cols, feature_val_cols, local_hour, KGClass, KGMajorClass, global_event_ID, lon, lat, time, base_value
    # we report data based on local_hour and KGMajorClass, so we need to group by local_hour and KGMajorClass and then sum the group_shap_cols andfeature_val_cols  

    # Calculate group SHAP values once, get DataFrames
    group_shap_df, group_feature_values_df, group_names = calculate_group_shap_values(
        shap_df, feature_values_df, feature_groups
    )
    logging.info(f"all_df columns: {all_df.columns.tolist()}, shape: {all_df.shape}")
    logging.info(f"shap_cols: {shap_cols}, shape: {shap_df.shape}")
    logging.info(f"feature_cols: {feature_cols}, shape: {feature_values_df.shape}")
    logging.info(
        f"group_shap_df columns: {group_shap_df.columns.tolist()}, shape: {group_shap_df.shape}"
    )
    logging.info(
        f"group_feature_values_df columns: {group_feature_values_df.columns.tolist()}, shape: {group_feature_values_df.shape}"
    )
    logging.info(f"group_names: {group_names}")

    # Generate summary plots if requested
    if args.summary_plots:
        generate_summary_and_kg_plots(
            shap_df,
            feature_values_df,
            feature_cols,
            output_dir,
            all_df,
            plot_type="feature",
        )
        generate_summary_and_kg_plots(
            group_shap_df,
            group_feature_values_df,
            group_names,
            output_dir,
            all_df,
            plot_type="group",
        )
        return

    # Generate day/night summary if requested
    if args.day_night_summary:
        summary_df = create_day_night_summary(group_shap_df, all_df, output_dir)
        logging.info("\nFeature Group Day/Night Summary:")
        logging.info("\n" + str(summary_df))
        return


    raw_df = all_df
    # base_value is the different for each hour, so we need to get the base_value for each hour 
    base_values: pd.Series = raw_df.groupby("local_hour")["base_value"].first()

    # group all_df by local_hour and KGMajorClass, exclude  "global_event_ID",  "lon",  "lat",  "time", "KGClass",
    exclude_cols = ["global_event_ID", "lon", "lat", "time", "KGClass"]
    #drop the exclude_cols then group by local_hour and KGMajorClass    
    all_df_by_hour_kg = raw_df.drop(columns=exclude_cols).groupby(["local_hour", "KGMajorClass"]).sum().reset_index()
    # include ["local_hour", "KGMajorClass"] in the shap_cols and feature_cols  
    shap_df = all_df_by_hour_kg[shap_cols]
    feature_values_df = all_df_by_hour_kg[feature_cols]
    
    # we report data based on local_hour and KGMajorClass, so we need to group by local_hour and KGMajorClass and then sum the shap_cols and feature_val_cols

    # the feature group data looks like this: please note there is not group_feature_val_cols, it is just feature_val_cols  
    # group_shap_cols, feature_val_cols, local_hour, KGClass, KGMajorClass, global_event_ID, lon, lat, time, base_value
    # we report data based on local_hour and KGMajorClass, so we need to group by local_hour and KGMajorClass and then sum the group_shap_cols andfeature_val_cols  

    # Calculate group SHAP values once, get DataFrames. Please note the group_feature_values_df is just a direct sum of feature values, for the feature in the group
    group_shap_df, group_feature_values_df, group_names = calculate_group_shap_values(
        shap_df, feature_values_df, feature_groups
    )
    
    # add local_hour and KGMajorClass to the group_shap_df and group_feature_values_df
    group_shap_df["local_hour"] = all_df_by_hour_kg["local_hour"]
    group_shap_df["KGMajorClass"] = all_df_by_hour_kg["KGMajorClass"]
    group_feature_values_df["local_hour"] = all_df_by_hour_kg["local_hour"]
    group_feature_values_df["KGMajorClass"] = all_df_by_hour_kg["KGMajorClass"] 

    logging.info(f"all_df_by_hour_kg columns: {all_df_by_hour_kg.columns.tolist()}, shape: {all_df_by_hour_kg.shape}")
    logging.info(f"shap_cols: {shap_cols}, shape: {shap_df.shape}")
    logging.info(f"feature_cols: {feature_cols}, shape: {feature_values_df.shape}")
    logging.info(
        f"group_shap_df columns: {group_shap_df.columns.tolist()}, shape: {group_shap_df.shape}"
    )
    logging.info(
        f"group_feature_values_df columns: {group_feature_values_df.columns.tolist()}, shape: {group_feature_values_df.shape}"
    )
    logging.info(f"group_names: {group_names}")


    # Load and prepare feature values for plotting
    feature_values_melted = load_feature_values(shap_values_feather_path)
    # Add local_hour from all_df
    # group_shap_df["local_hour"] = all_df_by_hour_kg["local_hour"]
    # feature_values_df["local_hour"] = all_df_by_hour_kg["local_hour"]

    # Generate plots for each climate zone
    kg_classes = ["global"] + all_df_by_hour_kg["KGMajorClass"].unique().tolist()
    for kg_class in kg_classes:
        # Prepare data for current climate zone
        feature_group_data = prepare_feature_group_data(group_shap_df, kg_class)
        feature_group_data = feature_group_data.drop(columns=["KGMajorClass"])

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
        plot_shap_and_feature_values(
            df_feature=shap_df,
            feature_values_melted=feature_values_melted,
            kg_classes=[kg_class],
            output_dir=output_dir,
            base_value=base_values,
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
        help="Only generate the day/night summary without creating other plots.",
    )
    parser.add_argument(
        "--summary-plots",
        action="store_true",
        default=False,
        help="Only generate the summary plots then exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
