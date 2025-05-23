import logging
import os
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .group_data import GroupData

from .plot_shap_stacked_bar import plot_shap_stacked_bar, _save_plot_data
from .plot_util import (
    get_feature_groups,
    get_latex_label,
    replace_cold_with_continental,
    FEATURE_COLORS,
    get_label_with_unit,
    get_unit,
    get_long_name_without_unit,
)

def generate_group_shap_plots_by_climate_zone(
    obj_group_data: GroupData,
    kg_classes: List[str],
    output_dir: str,
    base_values: pd.Series,
    show_total_feature_line: bool = True,
) -> None:
    """
    Generates SHAP value contribution plots for feature groups across different climate zones.
    Creates both global and per-climate-zone visualizations of group-level SHAP contributions.

    Args:
        obj_group_data: GroupData object containing processed data
        kg_classes: List of KGMajorClasses
        output_dir: Directory to save output
        base_values: Base values for SHAP contributions
        show_total_feature_line: Whether to show total feature value line (default: True)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Use the predefined color mapping
    color_mapping = FEATURE_COLORS

    # Create a global subdirectory
    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)

    # Generate and save the standalone SHAP stacked bar plot for global data (all features)
    print("Generating global standalone SHAP stacked bar plot (all features)...")
    shap_plot_df_global = obj_group_data.shap_group_hourly_mean_df("global")

    # Plot and save the standalone SHAP stacked bar plot for global data
    plot_shap_stacked_bar(
        shap_df=shap_plot_df_global,
        title="Global SHAP Value Contributions (All Features)",
        output_path=os.path.join(
            global_dir, "global_shap_stacked_bar_all_features.png"
        ),
        color_mapping=color_mapping,
    )

    # Plot for each KGMajorClass
    for kg_class in kg_classes:
        print(f"Generating plots for KGMajorClass '{kg_class}'...")
        # Create subdirectory for the current kg_class
        kg_class_dir = os.path.join(output_dir, kg_class)
        os.makedirs(kg_class_dir, exist_ok=True)

        # Get SHAP data for the current kg_class
        shap_plot_df_kg = obj_group_data.shap_group_hourly_mean_df(kg_class)
        
        # Check if there's data for the current kg_class
        if shap_plot_df_kg.empty:
            print(f"No data for KGMajorClass '{kg_class}'. Skipping.")
            continue

        plot_shap_stacked_bar(
            shap_df=shap_plot_df_kg,
            title=f"SHAP Value Contributions (All Features) - KGMajorClass {kg_class}",
            output_path=os.path.join(
                kg_class_dir, f"{kg_class}_shap_stacked_bar_all_features.png"
            ),
            color_mapping=color_mapping,
        )

        # Generate side-by-side plots for each feature group
        for group_name in obj_group_data.group_names:
            # Get feature data for this group. Its columns are the features in this group.
            feature_values_plot_df = obj_group_data.feature_hourly_mean_for_a_given_group_df(kg_class, group_name)

            if feature_values_plot_df.empty:
                logging.info(f"No feature data for group '{group_name}' in KGMajorClass '{kg_class}'. Side-by-side SHAP plot will be empty.")
                # Create an empty shap_df with same index and columns as the (empty) feature_values_plot_df
                shap_plot_df = pd.DataFrame(index=feature_values_plot_df.index, columns=feature_values_plot_df.columns)
            else:
                features_in_group = feature_values_plot_df.columns
                all_shap_df_for_kg_class = obj_group_data.shap_group_hourly_mean_df(kg_class) # This is shap_plot_df_kg

                # Select SHAP values for features present in the current group AND in the main SHAP DataFrame
                columns_to_select_for_shap = [f for f in features_in_group if f in all_shap_df_for_kg_class.columns]

                if not columns_to_select_for_shap:
                    logging.warning(f"No SHAP data found for any features in group '{group_name}' (features: {list(features_in_group)}) for KGMajorClass '{kg_class}'. SHAP values will be plotted as zero.")
                    # Create a DataFrame of zeros, indexed like all_shap_df_for_kg_class, columns like features_in_group
                    shap_plot_df = pd.DataFrame(0, index=all_shap_df_for_kg_class.index, columns=features_in_group)
                else:
                    shap_plot_df = all_shap_df_for_kg_class[columns_to_select_for_shap]
                    # Ensure shap_plot_df has all columns from features_in_group, filling missing SHAP values with 0.
                    # This makes shap_df.columns identical to feature_values_df.columns.
                    shap_plot_df = shap_plot_df.reindex(columns=features_in_group, fill_value=0)

                if len(columns_to_select_for_shap) < len(features_in_group):
                    missing_shap_features = set(features_in_group) - set(columns_to_select_for_shap)
                    logging.warning(f"SHAP data missing for some features in group '{group_name}': {missing_shap_features}. Their SHAP contributions will be shown as 0.")

            # --- Ensure consistent x-axis (local hour) for both plots ---
            common_index = feature_values_plot_df.index.intersection(shap_plot_df.index)
            shap_plot_df = shap_plot_df.reindex(index=common_index)
            feature_values_plot_df = feature_values_plot_df.reindex(index=common_index)
            
            # Determine the base value for the plot, defaulting to 0.0 if not found or incorrect type
            base_val_for_plot = 0.0
            if isinstance(base_values, pd.Series):
                base_val_for_plot = base_values.get(kg_class, 0.0)
            elif isinstance(base_values, (int, float)):
                base_val_for_plot = float(base_values)

            create_side_by_side_group_plot(
                shap_df=shap_plot_df,
                feature_values_df=feature_values_plot_df,
                group_name=group_name,
                output_dir=kg_class_dir,
                kg_class=kg_class,
                color_mapping=color_mapping,
                base_value=base_val_for_plot,
                show_total_feature_line=show_total_feature_line,
                x_axis_label="Local Hour",
            )


def create_side_by_side_group_plot(
    shap_df: pd.DataFrame,
    feature_values_df: pd.DataFrame,
    group_name: str,
    output_dir: str,
    kg_class: str,
    color_mapping: Dict[str, str],
    base_value: float = 0,
    show_total_feature_line: bool = True,
    x_axis_label: str = "Local Hour",
) -> None:
    """
    Creates a side-by-side visualization comparing SHAP contributions and actual values for a feature group.
    Left plot shows stacked SHAP values, right plot shows the actual feature values within the group.

    Args:
        shap_df: DataFrame containing SHAP values for the feature group
        feature_values_df: DataFrame containing feature values for the group
        group_name: Name of the feature group
        output_dir: Directory to save output
        kg_class: KGMajorClass name
        color_mapping: Dictionary mapping features to colors
        base_value: Base value for SHAP contributions (default: 0)
        show_total_feature_line: Whether to show total feature value line (default: True)
        x_axis_label: Label for the x-axis (default: "Local Hour")
    """
    import matplotlib.pyplot as plt
    import os

    plt.rcParams['text.usetex'] = False

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

    axes[0].set_title(f"SHAP Value Contributions")
    axes[0].set_xlabel(x_axis_label)
    axes[0].set_ylabel(f"{get_latex_label(group_name)} SHAP Contribution (°C)")

    # Calculate and plot mean SHAP values on the same axis
    mean_shap = mean_shap_df.sum(axis=1)  # Sum across features for each hour
    # Ensure mean_shap is plotted correctly even with NaNs that might come from sum() if all values are NaN
    # ax1.plot(mean_shap.index, mean_shap.values, color="black", marker="o", linewidth=2, label="Mean SHAP Contribution") # Removed as per user request

    # Get handles and labels for SHAP plot, convert feature names to LaTeX labels with units
    handles, labels = axes[0].get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label.startswith("Mean SHAP") or label.startswith("Base Value"):
            new_labels.append(label)
        else:
            new_labels.append(get_label_with_unit(label))
    axes[0].legend(
        handles, new_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=6
    )

    # Plot feature values on axes[1]
    # Determine color(s) for feature plot
    if len(feature_values_df.columns) == 1:
        # Single feature group (group_name is the feature)
        # Use group_name for color lookup, assuming group_name is canonical
        plot_colors_for_features = [color_mapping.get(group_name, "#333333")]
    else:
        # Multi-feature group, colors based on individual feature column names in feature_values_df
        # These column names are assumed to be canonical for color_mapping
        plot_colors_for_features = [color_mapping.get(col, "#333333") for col in feature_values_df.columns]

    feature_values_df.plot(ax=axes[1], color=plot_colors_for_features) # Apply determined colors
    
    axes[1].set_title(f"Feature Values - Group: {get_label_with_unit(group_name)}")
    axes[1].set_xlabel(x_axis_label)
    
    # Get unit for the feature group
    unit = get_unit(group_name)
    y_label = f"Feature Value{f' ({unit})' if unit else ''}"
    axes[1].set_ylabel(y_label)

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

    # Get handles and labels for feature values plot, convert feature names to LaTeX labels with units
    handles, labels = axes[1].get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label == "Total Feature Value":
            new_labels.append(label)
        else:
            new_labels.append(get_label_with_unit(label))
    axes[1].legend(
        handles, new_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5
    )

    # Adjust layout and save the figure
    group_name_formatted = get_long_name_without_unit(group_name)
    kg_class_formatted = replace_cold_with_continental(kg_class)

    # Prepare for mathtext bolding by escaping spaces to preserve them
    group_name_formatted_for_title = group_name_formatted.replace(' ', r'\ ')
    kg_class_formatted_for_title = kg_class_formatted.replace(' ', r'\ ')

    # Construct the title using LaTeX for bold text and preserving spaces
    # Escaping curly braces for f-string and LaTeX interaction
    # example r"This is a Normal and $\\bf{Bold}$ Title"
    title = f"HW-NHW UHI Contribution and Feature Values by Hour - $\\bf{{{group_name_formatted_for_title}}}$ - Climate Zone $\\bf{{{kg_class_formatted_for_title}}}$"
    print(title)
    plt.suptitle(title, y=1.02)
    
    # X-axis ticks for axes[1] to match axes[0]
    if not feature_values_df.empty:
        tick_values = feature_values_df.index
        axes[1].set_xticks(tick_values)
        try:
            # Ensure tick labels are simple integers if the index values are numeric hours
            axes[1].set_xticklabels([str(int(val)) for val in tick_values])
        except ValueError:
            # Fallback if index values are not convertible to int (e.g., already strings or other format)
            axes[1].set_xticklabels([str(val) for val in tick_values])

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
    _save_plot_data(shap_df, total_shap, output_path, "shap")
    _save_plot_data(feature_values_df, total_features, output_path, "feature")


def create_combined_plot(
    shap_df: pd.DataFrame,
    feature_values_df: pd.DataFrame,
    group_name: str,
    output_dir: str,
    kg_class: str,
    color_mapping: Dict[str, str],
    base_value: float = 0, # Base value is not directly used in this combined plot but kept for consistency
    show_total_feature_line: bool = True, # This might be re-evaluated based on how feature values are plotted
    x_axis_label: str = "Local Hour",
) -> None:
    """
    Creates a single plot with SHAP contributions and feature values, using two y-axes.
    SHAP contributions are on the left y-axis, and feature values are on the right y-axis.

    Args:
        shap_df: DataFrame containing SHAP values for the feature group
        feature_values_df: DataFrame containing feature values for the group
        group_name: Name of the feature group
        output_dir: Directory to save output
        kg_class: KGMajorClass name
        color_mapping: Dictionary mapping features to colors
        base_value: Base value for SHAP contributions (default: 0)
        show_total_feature_line: Whether to show total feature value line (default: True)
        x_axis_label: Label for the x-axis (default: "Local Hour")
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np # For handling potential NaN in mean_shap for plotting

    plt.rcParams['text.usetex'] = False

    if shap_df.empty or feature_values_df.empty:
        logging.warning(
            f"No data available for group '{group_name}' in KGMajorClass '{kg_class}' for combined plot. Skipping."
        )
        return

    group_dir = os.path.join(output_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)
    output_filename = f"combined_shap_and_feature_{group_name}_{kg_class}.png"
    output_path = os.path.join(group_dir, output_filename)

    fig, ax1 = plt.subplots(figsize=(12, 7)) # Single Axes object

    # Plot SHAP contributions on the first y-axis (ax1)
    shap_colors = [color_mapping.get(feature, "#333333") for feature in shap_df.columns]
    mean_shap_df = shap_df.copy()
    mean_shap_df.plot(kind="bar", stacked=True, ax=ax1, color=shap_colors, legend=False) # Legend handled later

    ax1.set_xlabel(x_axis_label)
    ax1.set_ylabel(f"{get_latex_label(group_name)} SHAP Contribution (°C)", color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')

    # Plot mean SHAP line
    mean_shap = mean_shap_df.sum(axis=1)
    # Ensure mean_shap is plotted correctly even with NaNs that might come from sum() if all values are NaN
    # ax1.plot(mean_shap.index, mean_shap.values, color="black", marker="o", linewidth=2, label="Mean SHAP Contribution") # Removed as per user request


    # Create a second y-axis for feature values, sharing the same x-axis
    ax2 = ax1.twinx()

    # Determine color(s) for feature plot
    if len(feature_values_df.columns) == 1:
        plot_colors_for_features = [color_mapping.get(group_name, "#333333")]
    else:
        plot_colors_for_features = [color_mapping.get(col, "#333333") for col in feature_values_df.columns]

    # Plot feature values on ax2
    # If multiple features, plot them individually. If single, it's just one line.
    for i, col in enumerate(feature_values_df.columns):
        ax2.plot(feature_values_df.index, feature_values_df[col], color=plot_colors_for_features[i], linestyle='-', marker='.', linewidth=1.5, label=get_label_with_unit(col))

    unit = get_unit(group_name)
    y_label_feature = f"{get_latex_label(group_name)} Value{f' ({unit})' if unit else ''}"
    ax2.set_ylabel(y_label_feature, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Add total feature values line if enabled and there are multiple features
    if show_total_feature_line and len(feature_values_df.columns) > 1:
        total_features = feature_values_df.sum(axis=1)
        ax2.plot(total_features.index, total_features.values, color="dimgray", linestyle='--', marker='x', linewidth=2, label="Total Feature Value")

    # --- Legend Handling ---
    # Collect handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels() # ax1 has no direct legendable items anymore after removing mean SHAP line, so handles1/labels1 might be empty.
    handles2, labels2 = ax2.get_legend_handles_labels() # For feature lines

    # Create labels for SHAP bars (since legend=False was used in bar plot)
    # We want one label per bar color/feature in SHAP
    shap_bar_labels = [f"{get_latex_label(col)} SHAP Contribution (°C)" for col in shap_df.columns]
    shap_bar_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping.get(feature, "#333333")) for feature in shap_df.columns]
    
    # Combine all handles and labels
    # Order: SHAP bars, Feature lines, Total Feature line (if exists)
    all_handles = shap_bar_handles + handles2 # handles1 from ax1 is no longer needed for mean SHAP line
    all_labels = shap_bar_labels + labels2   # labels1 from ax1 is no longer needed

    # Filter out duplicate labels if any, keeping the first occurrence
    unique_handles_labels = {}
    for handle, label in zip(all_handles, all_labels):
        if label not in unique_handles_labels:
            unique_handles_labels[label] = handle
    
    fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=max(1, len(unique_handles_labels) // 3))


    # Title and Layout
    group_name_formatted = get_long_name_without_unit(group_name)
    kg_class_formatted = replace_cold_with_continental(kg_class)
    group_name_formatted_for_title = group_name_formatted.replace(' ', r'\ ')
    kg_class_formatted_for_title = kg_class_formatted.replace(' ', r'\ ')

    title = f"Combined SHAP and Feature Values - $\\bf{{{group_name_formatted_for_title}}}$ - Climate Zone $\\bf{{{kg_class_formatted_for_title}}}$"
    plt.title(title, y=1.08) # Adjust y for suptitle if using subplots, else for single plot

    # Ensure x-axis ticks are displayed correctly (e.g., as integers for hours)
    if not shap_df.empty: # or feature_values_df, they share index
        tick_values = shap_df.index
        ax1.set_xticks(tick_values)
        try:
            ax1.set_xticklabels([str(int(val)) for val in tick_values])
        except ValueError:
            ax1.set_xticklabels([str(val) for val in tick_values])


    fig.tight_layout(rect=[0, 0.01, 1, 0.95]) # Adjust rect to make space for legend below
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(
        f"Combined plot saved as '{output_path}' for group '{group_name}' and KGMajorClass '{kg_class}'."
    )
