from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import logging
import argparse
import os

import sys
sys.path.append('/home/jguo/research/hw_global/ultimate/')

# Assuming plot_shap_stacked_bar.py and plot_util.py are in the same directory
from mlflow_tools.plot_shap_stacked_bar import plot_feature_group_stacked_bar, _save_plot_data
from mlflow_tools.plot_util import get_latex_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_feature_group_from_csv(csv_path: str, output_path: str, title: str, group_by_column: str, base_value_path: Optional[str] = None):
    """
    Plots a feature group stacked bar chart from a CSV file saved by _save_plot_data.

    Args:
        csv_path: Path to the CSV file containing plot data.
        output_path: Path to save the generated plot.
        title: Title of the plot.
        group_by_column: Column name to group by (e.g., 'local_hour').
        base_value_path: Optional path to a CSV file containing base values.
                         If None, base values are assumed to be zeros.
    """
    try:
        df = pd.read_csv(csv_path)

        # Check if 'base_value' column exists and use its first value if it does
        if 'base_value' in df.columns:
            base_values = pd.Series([df['base_value'].iloc[0]] * len(df.index), index=df.index)
        else:
            base_values = pd.Series([0.178] * len(df.index), index=df.index)

        # Remove the 'Total' column if it exists, as plot_feature_group_stacked_bar expects only feature group cols
        if 'Total' in df.columns:
            df = df.drop(columns=['Total'])

        # Assuming the index is the group_by_column
        df = df.set_index(df.columns[0])
        df.index.name = group_by_column

        plot_feature_group_stacked_bar(
            df=df.reset_index(), # plot_feature_group_stacked_bar expects group_by_column as a regular column
            group_by_column=group_by_column,
            output_path=output_path,
            title=title,
            base_values=base_values
        )
        logging.info(f"Feature group stacked bar plot created from '{csv_path}' and saved to '{output_path}'")

    except FileNotFoundError:
        logging.error(f"CSV file not found at '{csv_path}'. Please check the file path.")
    except Exception as e:
        logging.error(f"An error occurred while plotting: {e}")


def process_and_plot(csv_files, output_dir, title, group_by_column, base_value_path=None):
    for csv_file in csv_files:
        # Extract the name of the region from the file name
        region_name = os.path.splitext(os.path.basename(csv_file))[0].split('_')[-1]
        output_path = os.path.join(output_dir, f"Figure_7_stacked_bar_{region_name}.png")
        plot_feature_group_from_csv(
            csv_path=csv_file,
            output_path=output_path,
            title=f"{title} - {region_name}",
            group_by_column=group_by_column,
            base_value_path=base_value_path
        )


if __name__ == "__main__":
    csv_files = [
        '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/hourly_stacked_bar/adjusted_shap_values_Arid.csv',
        '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/hourly_stacked_bar/adjusted_shap_values_Cold.csv',
        '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/hourly_stacked_bar/adjusted_shap_values_Temperate.csv',
        '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/hourly_stacked_bar/adjusted_shap_values_Tropical.csv'
    ]
    output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_7'
    process_and_plot(
        csv_files=csv_files,
        output_dir=output_dir,
        title='SHAP Feature Importance by Hour',
        group_by_column='local_hour'
    )