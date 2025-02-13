import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import logging
import seaborn as sns

import sys
sys.path.append('/home/jguo/research/hw_global/ultimate/')
# Assuming plot_side_by_side.py and plot_util.py are in the same directory
from mlflow_tools.plot_side_by_side import create_side_by_side_group_plot
from mlflow_tools.plot_util import get_latex_label, replace_cold_with_continental

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_side_by_side_from_csv(
    shap_csv_path: str,
    feature_csv_path: str,
    output_dir: str,
    group_name: str,
    kg_class: str,
    show_total_feature_line: bool = True,
):
    """
    Plots a side-by-side group plot from SHAP and feature data CSV files.

    Args:
        shap_csv_path: Path to the CSV file containing SHAP data.
        feature_csv_path: Path to the CSV file containing feature data.
        output_dir: Directory to save the generated plot.
        group_name: Name of the feature group.
        kg_class: KGMajorClass name.
        show_total_feature_line: Whether to show total feature value line (default: True).
    """
    try:
        shap_df = pd.read_csv(shap_csv_path, index_col=0)
        feature_values_df = pd.read_csv(feature_csv_path, index_col=0)

        # Drop 'Total' column if it exists
        shap_df = shap_df.drop(columns=['Total'], errors='ignore')
        feature_values_df = feature_values_df.drop(columns=['Total'], errors='ignore')

        # Basic check to ensure dataframes are not empty
        if shap_df.empty:
            logging.warning(f"SHAP data CSV '{shap_csv_path}' is empty.")
            return
        if feature_values_df.empty:
            logging.warning(f"Feature data CSV '{feature_csv_path}' is empty.")
            return

        # Create a default color palette - adjust n_colors if you have more groups
        all_features = list(shap_df.columns) + list(feature_values_df.columns)
        unique_features = sorted(list(set(all_features)))
        palette = sns.color_palette("tab20", n_colors=len(unique_features))
        color_mapping = dict(zip(unique_features, palette))
    
        #only select the group_name column from shap_df
        shap_df = shap_df[[group_name]].copy()
        create_side_by_side_group_plot(
            shap_df=shap_df,
            feature_values_df=feature_values_df,
            group_name=group_name,
            output_dir=output_dir,
            kg_class=kg_class,
            color_mapping=color_mapping,
            show_total_feature_line=show_total_feature_line,
        )
        logging.info(f"Side-by-side plot created and saved in '{output_dir}'.")

    except FileNotFoundError:
        logging.error("CSV file not found. Please check the file paths.")
    except Exception as e:
        logging.error(f"An error occurred while plotting: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot side-by-side group plot from CSV data.")
    parser.add_argument("--shap_csv_path", default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/global_shap_stacked_bar_all_features_shap_data.csv', help="Path to the SHAP data CSV file.")
    parser.add_argument("--feature_csv_path", default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/shap_and_feature_values_U10_global_feature_data.csv', help="Path to the feature data CSV file.")
    parser.add_argument("--output_dir", default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper', help="Directory to save the output plot.")
    parser.add_argument("--group_name", default="U10", help="Name of the feature group.")
    parser.add_argument("--kg_class", default="global", help="KGMajorClass name.")
    parser.add_argument("--show_total_feature_line", action='store_true', help="Show total feature value line in feature plot.")
    parser.add_argument("--no_total_feature_line", dest='show_total_feature_line', action='store_false', help="Do not show total feature value line.")
    parser.set_defaults(show_total_feature_line=True)


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_side_by_side_from_csv(
        shap_csv_path=args.shap_csv_path,
        feature_csv_path=args.feature_csv_path,
        output_dir=args.output_dir,
        group_name=args.group_name,
        kg_class=args.kg_class,
        show_total_feature_line=args.show_total_feature_line,
    )