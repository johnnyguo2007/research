import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Optional, Dict, Tuple, List
from mlflow_tools.plot_util import get_latex_label, FEATURE_COLORS



def _save_plot_data(
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
    base_values: Optional[pd.Series] = None,
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
        base_value = base_values.iloc[0] if base_values is not None else 0
        shap_df.plot(kind="bar", stacked=True, color=colors, ax=ax, bottom=base_value)
    else:
        base_value = base_values.iloc[0] if base_values is not None else 0
        shap_df.plot(kind="bar", stacked=True, colormap="tab20", ax=ax, bottom=base_value)

    # Calculate mean SHAP values and add base_values
    base_value = base_values.iloc[0] if base_values is not None else 0
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
        y=base_value,  # Using the base_value we calculated above
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
    _save_plot_data(
        shap_df,  # Save original values without base_value adjustment
        mean_shap,
        output_path,
        "shap",
    )


def plot_feature_group_stacked_bar(
    df: pd.DataFrame, #has local_hour col but does not contain KGMajorClass col
    group_by_column: str,
    output_path: str,
    title: str,
    base_values: Optional[pd.Series] = None,
    color_mapping: Optional[Dict[str, str]] = None,
    save_data_dump: bool = True,
) -> None:
    """
    Plots a stacked bar chart of mean feature group contributions with mean SHAP value line.
    df: dataframe containing group shap values with local_hour col
    group_shap_cols: list of group SHAP value column names
    """
    # Get group SHAP columns (all columns except local_hour)
    group_shap_cols = [col for col in df.columns if col != group_by_column]
    
    # Group by the specified column and compute means
    pivot_df = df.groupby(group_by_column)[group_shap_cols].mean()

    # Calculate means including base_values
    mean_values = pivot_df.sum(axis=1) + base_values

    # Save data before plotting, if requested
    if save_data_dump:
        _save_plot_data(pivot_df, mean_values, output_path, "group")

    # Sort the index if necessary
    pivot_df = pivot_df.sort_index()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use color_mapping if provided
    plot_colors = None
    if color_mapping:
        # Ensure columns are sorted consistently if needed, or map directly
        # Assuming pivot_df columns match keys in color_mapping (after potential suffix removal)
        try:
            plot_colors = [color_mapping[col.replace('_shap', '')] for col in pivot_df.columns]
        except KeyError as e:
            logging.warning(f"Color key not found: {e}. Falling back to default colormap.")
            plot_colors = None # Fallback if a key is missing

    # Plot stacked bars starting from base_values
    pivot_df.plot(kind="bar", stacked=True, ax=ax, bottom=base_values, color=plot_colors, colormap=None if plot_colors else 'tab20')

    # Plot mean values including base_values for feature group reports
    mean_values = pivot_df.sum(axis=1) + base_values
    mean_values.plot(
        color="black", marker="o", linewidth=2, ax=ax, label="Mean SHAP + Base Value"
    )

    # Add base value line
    ax.axhline(
        y=base_values.iloc[0],  # Assuming the first base value for the line
        color="red",
        linestyle="--",
        label=f"Base Value ({base_values.iloc[0]:.3f})",
    )

    # Get handles and labels, convert feature group names to LaTeX labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label.startswith("Mean SHAP") or label.startswith("Base Value"):
            new_labels.append(label)
        else:
            # Remove the '_shap' suffix from the group name before converting to LaTeX label
            clean_label = label.replace('_shap', '')
            new_labels.append(get_latex_label(clean_label))

    # Adjust legend position for potentially wider labels
    ax.legend(handles, new_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

    plt.title(title)
    ax.set_xlabel(group_by_column.replace("_", " ").title())
    ax.set_ylabel("Mean SHAP Value Contribution")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

