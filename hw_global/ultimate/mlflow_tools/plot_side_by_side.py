import logging
import os
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .plot_shap_stacked_bar import plot_shap_stacked_bar
from .plot_util import get_feature_groups, get_latex_label, replace_cold_with_continental


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
        df_feature: DataFrame containing feature data with features as columns
        feature_values_melted: Melted DataFrame containing feature values
        kg_classes: List of KGMajorClasses
        output_dir: Directory to save output
        base_value: Base value for SHAP contributions (default: 0)
        show_total_feature_line: Whether to show total feature value line (default: True)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get feature names (all columns except local_hour and KGMajorClass)
    feature_names = [
        col for col in df_feature.columns if col not in ["local_hour", "KGMajorClass"]
    ]
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
    shap_plot_df_global = df_feature.set_index("local_hour")[feature_names]

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
        # Get features in the current group
        group_features = [f for f, g in feature_groups.items() if g == group_name]

        # Select only the features for this group
        shap_plot_df_group = df_feature.set_index("local_hour")[group_features]

        # Get corresponding feature values
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
        )

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
        if kg_class == "global":
            continue

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
        shap_plot_df_kg = df_feature_subset.set_index("local_hour")[feature_names]

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
            # Get features in the current group
            group_features = [f for f, g in feature_groups.items() if g == group_name]

            # Select only the features for this group
            shap_plot_df = df_feature_subset.set_index("local_hour")[group_features]

            # Get corresponding feature values
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
            )

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

