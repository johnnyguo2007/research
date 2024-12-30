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
from mlflow_tools.plot_summary import generate_summary_and_kg_plots
from mlflow_tools.create_night_day_summary import create_day_night_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_feature_groups(feature_names: List[str]) -> Dict[str, str]:
    """
    Assign features to groups based on specified rules.

    Args:
        feature_names (list): List of feature names.

    Returns:
        dict: Mapping from feature names to group names.
    """
    prefixes = ("delta_", "hw_nohw_diff_", "Double_Differencing_")
    feature_groups = {}
    for feature in feature_names:
        group = feature
        for prefix in prefixes:
            if feature.startswith(prefix):
                group = feature[len(prefix) :]
                break
        # If feature does not start with any prefix, it is its own group, but name the group feature + "Level"
        if group == feature:
            group = feature + "_Level"
        feature_groups[feature] = group
    return feature_groups


def replace_cold_with_continental(kg_main_group: str) -> str:
    """
    Replaces 'Cold' with 'Continental' in the given string.

    Args:
        kg_main_group (str): The input string.

    Returns:
        str: The modified string with 'Cold' replaced by 'Continental'.
    """
    if kg_main_group == "Cold":
        return "Continental"
    return kg_main_group


# Add lookup table reading
lookup_df = pd.read_excel(
    "/home/jguo/research/hw_global/Data/var_name_unit_lookup.xlsx"
)
lookup_dict = dict(zip(lookup_df["Variable"], lookup_df["LaTeX"]))


def get_latex_label(feature_name: str) -> str:
    """
    Retrieves the LaTeX label for a given feature based on its feature group.

    Args:
        feature_name (str): The name of the feature.

    Returns:
        str: The corresponding LaTeX label.
    """
    # Define mapping from prefixes to symbols
    prefix_to_symbol = {
        "delta_": "(Δ)",
        "hw_nohw_diff_": "HW-NHW ",
        "Double_Differencing_": "(Δ)HW-NHW ",
    }
    symbol = ""
    feature_group = feature_name
    for prefix in prefix_to_symbol.keys():
        if feature_name.startswith(prefix):
            feature_group = feature_name[len(prefix) :]
            symbol = prefix_to_symbol[prefix]
            break
    # if feature_group == feature_name:
    #     feature_group += "_Level"

    # Get the LaTeX label from the lookup dictionary
    latex_label = lookup_dict.get(feature_group)

    # Use the original feature group if LaTeX label is not found
    if pd.isna(latex_label) or latex_label == "":
        latex_label = feature_group
    # Combine symbol and LaTeX label
    final_label = f"{symbol}{latex_label}".strip()
    return final_label


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


def save_plot_data(
    df: pd.DataFrame, total_values: pd.Series, output_path: str, plot_type: str
) -> None:
    """
    Save plot data to CSV files, including both individual values and totals.

    Args:
        df: DataFrame containing the plot data
        total_values: Series containing the total values
        output_path: Base path for the output file
        plot_type: String indicating the type of plot ('shap' or 'feature')
    """
    # Remove .png extension and add csv
    base_path = output_path.rsplit(".", 1)[0]

    # Create a copy of the DataFrame
    output_df = df.copy()

    # Add the total values as a new column
    output_df["Total"] = total_values

    # Save to CSV
    output_df.to_csv(f"{base_path}_{plot_type}_data.csv")

    logging.info(f"Saved {plot_type} data to {base_path}_{plot_type}_data.csv")


def plot_shap_stacked_bar(
    shap_df: pd.DataFrame,
    title: str,
    output_path: str,
    color_mapping: Optional[Dict[str, str]] = None,
    return_fig: bool = False,
    base_value: float = 0,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plots a standalone SHAP stacked bar plot (all features) with a mean SHAP value curve.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    if color_mapping:
        sorted_columns = sorted(shap_df.columns, key=lambda x: x)
        colors = [color_mapping.get(feature, "#333333") for feature in sorted_columns]
        shap_df = shap_df[sorted_columns]
        shap_df.plot(kind="bar", stacked=True, color=colors, ax=ax, bottom=base_value)
    else:
        shap_df.plot(
            kind="bar", stacked=True, colormap="tab20", ax=ax, bottom=base_value
        )

    # Calculate mean SHAP values and add base_value
    mean_shap = shap_df.sum(axis=1) + base_value

    # Plot the mean SHAP values as a line on the same axis
    mean_shap.plot(
        kind="line",
        color="black",
        marker="o",
        linewidth=2,
        ax=ax,
        label="Mean SHAP + Base Value",
    )

    # Add base value line
    ax.axhline(
        y=base_value,
        color="red",
        linestyle="--",
        label=f"Base Value ({base_value:.3f})",
    )

    # Get handles and labels, convert feature names to LaTeX labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label.startswith("Mean SHAP") or label.startswith("Base Value"):
            new_labels.append(label)
        else:
            new_labels.append(get_latex_label(label))

    ax.legend(
        handles, new_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=6
    )

    plt.title(title)
    plt.xlabel("Hour of Day")
    ax.set_ylabel("Mean SHAP Value Contribution")
    plt.tight_layout()

    if return_fig:
        return fig, ax
    else:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logging.info(
            f"Standalone SHAP stacked bar plot with mean curve saved at '{output_path}'."
        )

    # Save data before plotting
    save_plot_data(
        shap_df,  # Save original values without base_value adjustment
        mean_shap,
        output_path,
        "shap",
    )


def plot_shap_and_feature_values_for_group(
    shap_df: pd.DataFrame,
    feature_values_df: pd.DataFrame,
    group_name: str,
    output_dir: str,
    kg_class: str,
    color_mapping: Dict[str, str],
    base_value: float = 0,
    show_total_feature_line: bool = True,
) -> None:
    """
    Plots SHAP value contributions and feature group's values side by side.

    Args:
        shap_df: DataFrame containing SHAP values
        feature_values_df: DataFrame containing feature values
        group_name: Name of the feature group
        output_dir: Directory to save output
        kg_class: KGMajorClass name
        color_mapping: Dictionary mapping features to colors
        base_value: Base value for SHAP contributions (default: 0)
        show_total_feature_line: Whether to show total feature value line (default: True)
    """
    import matplotlib.pyplot as plt
    import os

    # Check if data is available
    if shap_df.empty or feature_values_df.empty:
        logging.warning(
            f"No data available for group '{group_name}' in KGMajorClass '{kg_class}'. Skipping."
        )
        return

    # Create a subdirectory for the current feature group if it doesn't exist
    group_dir = os.path.join(output_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)

    # Prepare the output paths
    output_filename = f"shap_and_feature_values_{group_name}_{kg_class}.png"
    output_path = os.path.join(group_dir, output_filename)

    shap_output_filename = f"shap_contributions_{group_name}_{kg_class}.png"
    shap_output_path = os.path.join(group_dir, shap_output_filename)

    # Plot SHAP contributions using plot_shap_stacked_bar (standalone plot)
    plot_shap_stacked_bar(
        shap_df=shap_df,
        title=f"SHAP Value Contributions - {group_name} - KGMajorClass {kg_class}",
        output_path=shap_output_path,
        color_mapping=color_mapping,
        return_fig=False,
    )

    # Plot SHAP contributions and feature values for combined plot
    fig_combined, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8), sharex=False)

    # Define colors based on the color mapping
    shap_colors = [color_mapping.get(feature, "#333333") for feature in shap_df.columns]
    feature_colors = [
        color_mapping.get(feature, "#333333") for feature in feature_values_df.columns
    ]

    # Plot SHAP contributions on axes[0]
    # Calculate mean SHAP values for each feature at each hour
    mean_shap_df = shap_df.copy()
    mean_shap_df.plot(kind="bar", stacked=True, ax=axes[0], color=shap_colors)

    axes[0].set_title("Mean SHAP Value Contributions")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Mean Contribution")

    # Calculate and plot mean SHAP values on the same axis
    mean_shap = mean_shap_df.sum(axis=1)  # Sum across features for each hour
    mean_shap.plot(kind="line", color="black", marker="o", linewidth=2, ax=axes[0])

    # Get handles and labels for SHAP plot, convert feature names to LaTeX labels
    handles, labels = axes[0].get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label.startswith("Mean SHAP") or label.startswith("Base Value"):
            new_labels.append(label)
        else:
            new_labels.append(get_latex_label(label))
    axes[0].legend(
        handles, new_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=6
    )

    # Plot feature values on axes[1]
    feature_values_df.plot(ax=axes[1], color=feature_colors)
    axes[1].set_title(f"Feature Values - Group: {get_latex_label(group_name)}")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Feature Value")
    axes[1].axhline(0, linestyle="--", color="lightgray", linewidth=1)

    # Add total feature values line if enabled and there are multiple features
    if show_total_feature_line and len(feature_values_df.columns) > 1:
        # Calculate and plot total feature values on the same axis
        total_features = feature_values_df.sum(axis=1)
        total_features.plot(
            kind="line",
            color="black",
            marker="o",
            linewidth=2,
            ax=axes[1],
            label="Total Feature Value",
        )

    # Get handles and labels for feature values plot, convert feature names to LaTeX labels
    handles, labels = axes[1].get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label == "Total Feature Value":
            new_labels.append(label)
        else:
            new_labels.append(get_latex_label(label))
    axes[1].legend(
        handles, new_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5
    )

    # Adjust layout and save the figure
    title = f"HW-NHW UHI Contribution and Feature Values by Hour - {get_latex_label(group_name)} - Climate Zone {replace_cold_with_continental(kg_class)}"
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logging.info(
        f"Plot saved as '{output_path}' for group '{group_name}' and KGMajorClass '{kg_class}'."
    )

    # Calculate totals
    total_shap = shap_df.sum(axis=1)
    total_features = (
        feature_values_df.sum(axis=1)
        if len(feature_values_df.columns) > 1
        else pd.Series(0, index=feature_values_df.index)
    )

    # Save data before plotting
    save_plot_data(shap_df, total_shap, output_path, "shap")
    save_plot_data(feature_values_df, total_features, output_path, "feature")


def plot_shap_and_feature_values(
    df_feature: pd.DataFrame,
    feature_values_melted: pd.DataFrame,
    kg_classes: List[str],
    output_dir: str,
    base_value: float = 0,
    show_total_feature_line: bool = True,
) -> None:
    """
    Plots SHAP value contributions and feature group's values side by side.

    Args:
        df_feature: DataFrame containing feature data
        feature_values_melted: Melted DataFrame containing feature values
        kg_classes: List of KGMajorClasses
        output_dir: Directory to save output
        base_value: Base value for SHAP contributions (default: 0)
        show_total_feature_line: Whether to show total feature value line (default: True)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Prepare a list of all feature groups
    feature_names = df_feature["Feature"].unique().tolist()
    feature_groups = get_feature_groups(feature_names)
    unique_groups = set(feature_groups.values())

    # Create a color palette
    palette = sns.color_palette("tab20", n_colors=len(feature_names))
    color_mapping = dict(zip(sorted(feature_names), palette))

    # Create a global subdirectory
    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)

    # Generate and save the standalone SHAP stacked bar plot for global data (all features)
    print("Generating global standalone SHAP stacked bar plot (all features)...")
    shap_plot_df_global = df_feature.pivot_table(
        index="local_hour", columns="Feature", values="Value", fill_value=0
    ).reset_index()
    shap_plot_df_global.set_index("local_hour", inplace=True)

    # Plot and save the standalone SHAP stacked bar plot for global data
    plot_shap_stacked_bar(
        shap_df=shap_plot_df_global,
        title="Global SHAP Value Contributions (All Features)",
        output_path=os.path.join(
            global_dir, "global_shap_stacked_bar_all_features.png"
        ),
        color_mapping=color_mapping,
    )

    # Generate side-by-side plots for each feature group in global data
    print("Generating global plots for each feature group...")
    for group_name in unique_groups:
        # Filter df_feature for features in the current group
        group_features = [f for f, g in feature_groups.items() if g == group_name]

        shap_plot_df_group = df_feature[df_feature["Feature"].isin(group_features)]
        shap_plot_df_group = (
            shap_plot_df_group.groupby(["local_hour", "Feature"])["Value"]
            .sum()
            .reset_index()
        )
        shap_plot_df_group = shap_plot_df_group.pivot_table(
            index="local_hour", columns="Feature", values="Value", fill_value=0
        ).reset_index()
        shap_plot_df_group.set_index("local_hour", inplace=True)

        feature_values_plot_df_group = feature_values_melted[
            feature_values_melted["Feature"].isin(group_features)
        ]
        feature_values_plot_df_group = (
            feature_values_plot_df_group.groupby(["local_hour", "Feature"])[
                "FeatureValue"
            ]
            .mean()
            .reset_index()
        )
        feature_values_plot_df_group = feature_values_plot_df_group.pivot_table(
            index="local_hour", columns="Feature", values="FeatureValue", fill_value=0
        ).reset_index()
        feature_values_plot_df_group.set_index("local_hour", inplace=True)

        plot_shap_and_feature_values_for_group(
            shap_df=shap_plot_df_group,
            feature_values_df=feature_values_plot_df_group,
            group_name=group_name,
            output_dir=global_dir,
            kg_class="global",
            color_mapping=color_mapping,
            base_value=base_value,
            show_total_feature_line=show_total_feature_line,
        )

    # Plot for each KGMajorClass
    for kg_class in kg_classes:
        print(f"Generating plots for KGMajorClass '{kg_class}'...")
        # Create subdirectory for the current kg_class
        kg_class_dir = os.path.join(output_dir, kg_class)
        os.makedirs(kg_class_dir, exist_ok=True)

        # Filter data for the current kg_class
        df_feature_subset = df_feature[df_feature["KGMajorClass"] == kg_class]
        feature_values_subset = feature_values_melted[
            feature_values_melted["KGMajorClass"] == kg_class
        ]

        # Check if there's data for the current kg_class
        if df_feature_subset.empty:
            print(f"No data for KGMajorClass '{kg_class}'. Skipping.")
            continue

        # Generate and save the standalone SHAP stacked bar plot for the current kg_class (all features)
        print(
            f"Generating standalone SHAP stacked bar plot for KGMajorClass '{kg_class}'..."
        )
        shap_plot_df_kg = df_feature_subset.pivot_table(
            index="local_hour", columns="Feature", values="Value", fill_value=0
        ).reset_index()
        shap_plot_df_kg.set_index("local_hour", inplace=True)

        plot_shap_stacked_bar(
            shap_df=shap_plot_df_kg,
            title=f"SHAP Value Contributions (All Features) - KGMajorClass {kg_class}",
            output_path=os.path.join(
                kg_class_dir, f"{kg_class}_shap_stacked_bar_all_features.png"
            ),
            color_mapping=color_mapping,
        )

        # Generate side-by-side plots for each feature group
        for group_name in unique_groups:
            # Filter df_feature_subset for features in the current group
            group_features = [f for f, g in feature_groups.items() if g == group_name]

            shap_plot_df = df_feature_subset[
                df_feature_subset["Feature"].isin(group_features)
            ]
            shap_plot_df = (
                shap_plot_df.groupby(["local_hour", "Feature"])["Value"]
                .sum()
                .reset_index()
            )
            shap_plot_df = shap_plot_df.pivot_table(
                index="local_hour", columns="Feature", values="Value", fill_value=0
            ).reset_index()
            shap_plot_df.set_index("local_hour", inplace=True)

            feature_values_plot_df = feature_values_subset[
                feature_values_subset["Feature"].isin(group_features)
            ]
            feature_values_plot_df = (
                feature_values_plot_df.groupby(["local_hour", "Feature"])[
                    "FeatureValue"
                ]
                .mean()
                .reset_index()
            )
            feature_values_plot_df = feature_values_plot_df.pivot_table(
                index="local_hour",
                columns="Feature",
                values="FeatureValue",
                fill_value=0,
            ).reset_index()
            feature_values_plot_df.set_index("local_hour", inplace=True)

            plot_shap_and_feature_values_for_group(
                shap_df=shap_plot_df,
                feature_values_df=feature_values_plot_df,
                group_name=group_name,
                output_dir=kg_class_dir,
                kg_class=kg_class,
                color_mapping=color_mapping,
                base_value=base_value,
                show_total_feature_line=show_total_feature_line,
            )


# Define the function with 'Value' instead of 'Importance'
def plot_feature_group_stacked_bar(
    df: pd.DataFrame,
    group_by_column: str,
    output_path: str,
    title: str,
    base_value: float = 0,
) -> None:
    """
    Plots a stacked bar chart of mean feature group contributions with mean SHAP value line.
    """
    # Pivot and prepare data - calculate mean instead of sum
    pivot_df = df.pivot_table(
        index=group_by_column,
        columns="Feature Group",
        values="Value",
        aggfunc="mean",
        fill_value=0,
    )

    # Calculate means including base_value
    mean_values = pivot_df.sum(axis=1) + base_value

    # Save data before plotting
    save_plot_data(pivot_df, mean_values, output_path, "group")

    # Sort the index if necessary
    pivot_df = pivot_df.sort_index()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot stacked bars starting from base_value
    pivot_df.plot(kind="bar", stacked=True, ax=ax, bottom=base_value)

    # Plot mean values including base_value for feature group reports
    mean_values = pivot_df.sum(axis=1) + base_value
    mean_values.plot(
        color="black", marker="o", linewidth=2, ax=ax, label="Mean SHAP + Base Value"
    )

    # Add base value line
    ax.axhline(
        y=base_value,
        color="red",
        linestyle="--",
        label=f"Base Value ({base_value:.3f})",
    )

    # Get handles and labels, convert feature group names to LaTeX labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label.startswith("Mean SHAP") or label.startswith("Base Value"):
            new_labels.append(label)
        else:
            new_labels.append(get_latex_label(label))

    ax.legend(handles, new_labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title(title)
    ax.set_xlabel(group_by_column.replace("_", " ").title())
    ax.set_ylabel("Mean SHAP Value Contribution")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


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

    return group_shap_df, group_feature_values_df, group_names


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


def prepare_data_for_plotting(
    df: pd.DataFrame, group_by_column: str, feature_groups: Dict[str, str]
) -> pd.DataFrame:
    """
    Prepares data for plotting by grouping and pivoting.

    Args:
        df (pd.DataFrame): DataFrame containing feature data.
        group_by_column (str): Column to group by.
        feature_groups (dict): Mapping of features to groups.

    Returns:
        pd.DataFrame: Prepared DataFrame for plotting.
    """
    df_grouped = df.groupby([group_by_column, "Feature"])["Value"].sum().reset_index()
    df_pivot = df_grouped.pivot_table(
        index=group_by_column, columns="Feature", values="Value", fill_value=0
    )
    return df_pivot


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
    all_df = pd.read_feather(shap_values_feather_path)
    base_value = all_df["base_value"].iloc[0] if "base_value" in all_df.columns else 0

    # Get feature names and groups
    shap_cols, feature_cols = extract_shap_and_feature_columns(all_df)
    feature_groups = get_feature_groups(feature_cols)
    shap_df = all_df[shap_cols]
    feature_values_df = all_df[feature_cols]

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

    # Load and prepare feature values for plotting
    feature_values_melted = load_feature_values(shap_values_feather_path)

    # Generate plots for each climate zone
    kg_classes = ["global"] + all_df["KGMajorClass"].unique().tolist()
    for kg_class in kg_classes:
        # Prepare data for current climate zone
        feature_group_data = prepare_feature_group_data(group_shap_df, kg_class)

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
            feature_group_data, "local_hour", output_path, plot_title, base_value
        )

        # Generate SHAP and feature value plots
        plot_shap_and_feature_values(
            df_feature=feature_values_df,
            feature_values_melted=feature_values_melted,
            kg_classes=[kg_class],
            output_dir=output_dir,
            base_value=base_value,
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
