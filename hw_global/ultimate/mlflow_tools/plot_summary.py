import os
import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import List

from .group_data import GroupData
from .plot_util import get_latex_label

def _create_output_dir(output_dir: str, kg_class: str = None) -> str:
    """
    Creates the output directory if it doesn't exist.

    Args:
        output_dir (str): Base output directory.
        kg_class (str, optional): Name of the climate zone. Defaults to None.

    Returns:
        str: Path to the output directory.
    """
    if kg_class:
        output_dir = os.path.join(output_dir, kg_class)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
def get_shap_feature_importance(shap_values, feature_names):
    """
    Calculate the SHAP feature importance.
    """
    shap_feature_importance = np.abs(shap_values).mean(axis=0)
    total_importance = np.sum(shap_feature_importance)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_feature_importance,
        'Percentage': (shap_feature_importance / total_importance) * 100
    })
    shap_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    # shap_importance_df = add_long_name(shap_importance_df, join_column='Feature', df_daily_vars=df_daily_vars)
    return shap_importance_df

def plot_summary(
    shap_values_df: pd.DataFrame,
    feature_values_df: pd.DataFrame,
    feature_names: List[str],
    output_dir: str,
    kg_class: str = "Global",
    plot_type: str = "feature"
) -> None:
    """
    Generate SHAP summary plot, either for individual features or feature groups.
    """
    logging.info(
        f"Starting plot_{plot_type}_summary for {kg_class} with shap_values shape: {shap_values_df.shape} and feature_values shape: {feature_values_df.shape}"
    )
    logging.info(f"shap_values_df columns: {shap_values_df.columns.tolist()}")
    logging.info(f"feature_values_df columns: {feature_values_df.columns.tolist()}")
    output_dir = _create_output_dir(output_dir)
    plt.figure()
    feature_names = [get_latex_label(name) for name in feature_names]
    shap_values = shap_values_df.values    # Calculate SHAP feature importance
    shap_feature_importance = get_shap_feature_importance(shap_values, feature_names)

    shap.summary_plot(
        shap_values[:, shap_feature_importance.index],
        feature_values_df.values[:, shap_feature_importance.index],
        feature_names=feature_names,
        show=False,
        plot_size=(20, 15),
        color_bar=True,
    )
    plt.title(f"{plot_type.capitalize()} Summary Plot - {kg_class}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{plot_type}_summary_plot_{kg_class}.png"))
    plt.close()

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_feature_importance['Percentage'].values,
            base_values=0,
            feature_names=feature_names
        ),
        show=False
    )
    plt.title(f"Feature Importance Plot - {kg_class}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_importance_plot_{kg_class}.png"))
    plt.close()

def generate_summary_and_kg_plots(
    group_data: GroupData,
    output_dir: str,
    plot_type: str
) -> None:
    """
    Generates summary plots for global data and each KGMajorClass using GroupData.

    Args:
        group_data (GroupData): GroupData object containing SHAP values and feature values.
        output_dir (str): Directory to save output plots.
        plot_type (str): Type of plot ('feature' or 'group').
    """
    summary_dir = _create_output_dir(os.path.join(output_dir, "summary_plots"))

    # Get appropriate data based on plot type
    if plot_type == 'feature':
        shap_df = group_data.shap_detail_df
        feature_values_df = group_data.feature_detail_df
        feature_names = group_data.feature_cols_names
    else:  # plot_type == 'group'
        shap_df = group_data.shap_group_detail_df
        feature_values_df = group_data.feature_group_detail_df
        feature_names = group_data.group_names

    feature_names = [get_latex_label(name) for name in feature_names]
    # Generate summary plots for global data
    logging.info(f"Generating {plot_type} summary plot for global data")
    plot_summary(
        shap_df, feature_values_df, feature_names, summary_dir, plot_type=plot_type
    )

    # Generate summary plots for each KGMajorClass
    kg_classes = group_data.df["KGMajorClass"].dropna().unique()
    for kg_class in kg_classes:
        kg_mask = group_data.df["KGMajorClass"] == kg_class
        logging.info(f"Generating {plot_type} summary plot for {kg_class}")
        kg_output_dir = _create_output_dir(summary_dir, kg_class)
        plot_summary(
            shap_df[kg_mask],
            feature_values_df[kg_mask],
            feature_names,
            kg_output_dir,
            kg_class,
            plot_type=plot_type
        )
