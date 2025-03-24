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

        # Remove the 'Total' column if it exists, as plot_feature_group_stacked_bar expects only feature group cols
        if 'Total' in df.columns:
            df = df.drop(columns=['Total'])

        # Assuming the index is the group_by_column
        df = df.set_index(df.columns[0])
        df.index.name = group_by_column


        # Load base values if path is provided, otherwise use zeros
        if base_value_path:
            try:
                base_values_df = pd.read_csv(base_value_path, index_col=0) # Assuming index column contains the group_by_column values
                base_values = base_values_df['base_value'] # Assuming the base value column is named 'base_value'
                # Ensure base_values index matches the DataFrame index
                base_values = base_values.reindex(df.index, fill_value=0) # Fill with 0 for any missing indices
            except FileNotFoundError:
                logging.warning(f"Base value CSV file not found at '{base_value_path}'. Using default base value of 0.")
                base_values = pd.Series([0] * len(df.index), index=df.index)
            except KeyError:
                logging.error(f"Column 'base_value' not found in base value CSV. Using default base value of 0.")
                base_values = pd.Series([0] * len(df.index), index=df.index)
        else:
            logging.info("No base value CSV path provided. Using default base value of 0.")
            base_values = pd.Series([0.112] * len(df.index), index=df.index)


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot feature group stacked bar chart from CSV data.")
    parser.add_argument("--csv_path", 
                        # default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/global_shap_stacked_bar_all_features_shap_data.csv', 
                        # default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/hourly_stacked_bar/feature_group_contribution_by_hour_Arid_group_data.csv',
                        default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/hourly_stacked_bar/feature_group_contribution_by_hour_Tropical_group_data.csv',
                        help="Path to the input CSV data file.")
    parser.add_argument("--output_path", 
                        # default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_7/Figure_7_stacked_bar_through_hour_Arid.png', 
                        default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_7/Figure_7_stacked_bar_through_hour_Tropical.png', 
                        help="Path to save the output plot image.")
    parser.add_argument("--title", default='Tropical', help="Title of the plot.")
    parser.add_argument("--group_by_column", default="local_hour", help="Column to group by (default: local_hour).")
    parser.add_argument("--base_value_path", default=None, help="Optional path to CSV file containing base values.")

    args = parser.parse_args()

    plot_feature_group_from_csv(
        csv_path=args.csv_path,
        output_path=args.output_path,
        title=args.title,
        group_by_column=args.group_by_column,
        base_value_path=args.base_value_path
    )