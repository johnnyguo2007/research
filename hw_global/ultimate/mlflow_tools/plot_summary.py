import os
import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import List

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

def _plot_summary(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    plot_title: str,
    file_name: str,
) -> None:
    """
    Generates a SHAP summary plot.

    Args:
        shap_values (np.ndarray): Array containing SHAP values.
        feature_values (np.ndarray): Array containing feature values.
        feature_names (List[str]): List of feature names.
        output_dir (str): Directory to save the plot.
        plot_title (str): Title of the plot.
        file_name (str): Name of the output file.
    """
    if shap_values.size == 0 or feature_values.size == 0:
        logging.warning(f"No data available for {plot_title}. Skipping plot.")
        return

    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            feature_values,
            feature_names=feature_names,
            show=False,
            plot_size=(12, 8),
        )
        plt.title(plot_title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, file_name),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
        logging.info(f"Generated {plot_title}")
    except Exception as e:
        logging.error(f"Failed to generate {plot_title}: {str(e)}")
        plt.close()

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
    output_dir = _create_output_dir(output_dir)
    _plot_summary(
        shap_values_df.values,
        feature_values_df.values,
        feature_names,
        output_dir,
        f"{plot_type.capitalize()} Summary Plot - {kg_class}",
        f"{plot_type}_summary_plot_{kg_class}.png",
    )

def generate_summary_and_kg_plots(
    shap_df: pd.DataFrame,
    feature_values_df: pd.DataFrame,
    feature_names: List[str],
    output_dir: str,
    all_df: pd.DataFrame,
    plot_type: str
) -> None:
    """
    Generates summary plots for global data and each KGMajorClass.

    Args:
        shap_df (pd.DataFrame): DataFrame containing SHAP values.
        feature_values_df (pd.DataFrame): DataFrame containing feature values.
        feature_names (List[str]): List of feature names (or group names).
        output_dir (str): Directory to save output plots.
        all_df (pd.DataFrame): DataFrame containing all data.
        plot_type (str): Type of plot ('feature' or 'group').
    """
    summary_dir = _create_output_dir(os.path.join(output_dir, "summary_plots"))

    # Generate summary plots for global data
    logging.info(f"Generating {plot_type} summary plot for global data")
    plot_summary(
        shap_df, feature_values_df, feature_names, summary_dir, plot_type=plot_type
    )

    # Generate summary plots for each KGMajorClass
    if "KGMajorClass" in all_df.columns:
        for kg_class in all_df["KGMajorClass"].dropna().unique():
            kg_mask = all_df["KGMajorClass"] == kg_class
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
