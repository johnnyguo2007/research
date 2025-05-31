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
from mlflow_tools.plot_util import get_latex_label, FEATURE_COLORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


BASE_VALUE = 0.186


def plot_feature_group_from_csv(
    csv_path: str, 
    output_path: str, 
    title: str, 
    group_by_column: str, 
    base_value_path: Optional[str] = None,
    save_data_dump: bool = True  # New parameter
):
    """
    Plots a feature group stacked bar chart from a CSV file.

    Args:
        csv_path: Path to the CSV file containing plot data.
        output_path: Path to save the generated plot.
        title: Title of the plot.
        group_by_column: Column name to group by (e.g., 'local_hour').
        base_value_path: Optional path to a CSV file containing base values.
                         If None, base values are assumed to be based on BASE_VALUE.
        save_data_dump: Whether to save the processed data to a new CSV dump file. 
                          Should be False if reading from an existing dump.
    """
    try:
        df = pd.read_csv(csv_path)

        # Remove the 'Total' column if it exists, as plot_feature_group_stacked_bar expects only feature group cols
        if 'Total' in df.columns:
            df = df.drop(columns=['Total'])

        # Assuming the index is the group_by_column
        df = df.set_index(df.columns[0])
        df.index.name = group_by_column

        # Filter columns: keep only those that contain a substring from FEATURE_COLORS keys
        if not FEATURE_COLORS:
            logging.warning("FEATURE_COLORS is empty. Cannot filter columns based on it. Proceeding with all columns.")
        else:
            feature_color_keys = list(FEATURE_COLORS.keys()) # Ensure it's a list for 'in' check
            original_columns = df.columns.tolist()
            columns_to_keep = [
                col for col in original_columns
                if any(fc_key in col for fc_key in feature_color_keys)
            ]
            
            if len(columns_to_keep) < len(original_columns):
                columns_dropped = [col for col in original_columns if col not in columns_to_keep]
                logging.info(f"Filtered out columns based on FEATURE_COLORS keys. Dropped: {columns_dropped}")
            
            if not columns_to_keep:
                logging.warning(
                    f"No columns in the CSV at '{csv_path}' matched any key in FEATURE_COLORS. "
                    f"Available FEATURE_COLORS keys: {feature_color_keys}. "
                    f"Original columns in CSV: {original_columns}. "
                    "The plot will likely be empty or incorrect."
                )
            df = df[columns_to_keep]


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
            base_values = pd.Series([BASE_VALUE] * len(df.index), index=df.index)


        plot_feature_group_stacked_bar(
            df=df.reset_index(), # plot_feature_group_stacked_bar expects group_by_column as a regular column
            group_by_column=group_by_column,
            output_path=output_path,
            title=title,
            base_values=base_values,
            color_mapping=FEATURE_COLORS,
            save_data_dump=save_data_dump  # Pass the parameter through
        )
        logging.info(f"Feature group stacked bar plot created from '{csv_path}' and saved to '{output_path}'")

    except FileNotFoundError:
        logging.error(f"CSV file not found at '{csv_path}'. Please check the file path.")
    except Exception as e:
        logging.error(f"An error occurred while plotting: {e}")


if __name__ == "__main__":

    # Default base input directory parts, similar to figure_9_one_feature_side_by_side.py
    default_input_dir_base_tuple = (
        "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/",
        "summary/mlflow/mlartifacts/",
        "893793682234305734/67f2e168085e4507b0a79941b74d7eb7/",
        "artifacts/data_only_24_hourly/"
    )
    # Default base output directory
    default_output_base_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_8_hourly_stacked_plots/'

    parser = argparse.ArgumentParser(description="Plot feature group stacked bar chart from CSV data for multiple climate zones.")
    parser.add_argument("--input_dir_base", nargs='+', default=list(default_input_dir_base_tuple), 
                        help="Base parts of the input directory structure. Expects data in 'input_dir_base/climate_zone/filename'.")
    parser.add_argument("--output_dir", default=default_output_base_dir, 
                        help="Base directory to save the output plots. Plots will be saved in subdirectories named after climate zones.")
    parser.add_argument("--group_by_column", default="local_hour", 
                        help="Column to group by (default: local_hour).")
    parser.add_argument("--base_value_path", default=None, 
                        help="Optional path to a single CSV file containing base values, to be used for all climate zones. "
                             "If not provided, a default base value (BASE_VALUE constant) is used.")
    parser.add_argument("--read_from_plot_data_dump", action="store_true",
                        help="If set, reads data from CSVs previously saved alongside plots "
                             "(e.g., in 'output_dir/zone/hourly_stacked_bar_{zone}_data_for_plot.csv') "
                             "instead of the primary source data. The primary source data is typically from "
                             "'{input_dir_base}/{zone}/{zone}_group_shap_contribution_data.csv'. "
                             "Note: The dumped CSVs are saved *after* initial processing, including "
                             "filtering by FEATURE_COLORS, in the original run that created the dump.")

    args = parser.parse_args()

    climate_zones_to_process = ["global", "Arid", "Continental", "Temperate", "Tropical"]

    for zone in climate_zones_to_process:
        logging.info(f"--- Processing climate zone: {zone} ---")

        # Construct output plot path (needed for both modes to determine dump location if applicable)
        # Output structure: args.output_dir/climate_zone_name/hourly_stacked_bar_{climate_zone_name}.png
        output_dir_for_zone = os.path.join(args.output_dir, zone)
        os.makedirs(output_dir_for_zone, exist_ok=True)
        
        plot_base_filename = f"hourly_stacked_bar_{zone}"
        output_plot_path_for_zone = os.path.join(output_dir_for_zone, f"{plot_base_filename}.png")
        logging.info(f"Output plot will be saved to: {output_plot_path_for_zone}")

        # Determine the input CSV path based on the --read_from_plot_data_dump flag
        if args.read_from_plot_data_dump:
            # Option 1: Read from a previously dumped plot data CSV.
            # These files are saved by `_save_plot_data` (called by `plot_feature_group_stacked_bar`)
            # in the same directory as the plot, with a name like '{plot_filename_stem}_data_for_plot.csv'.
            # Correction: The actual name saved by _save_plot_data is '{plot_filename_stem}_group_data.csv'
            csv_path_for_zone = os.path.join(output_dir_for_zone, f"{plot_base_filename}_group_data.csv") # Corrected name
            logging.info(f"Attempting to read from plot data dump: {csv_path_for_zone}")
        else:
            # Option 2 (Default): Read from the primary source data.
            # Expected structure: /path_part1/path_part2/.../climate_zone_name/{climate_zone_name}_group_shap_contribution_data.csv
            input_zone_specific_dir = os.path.join(*args.input_dir_base, zone)
            csv_filename = f"{zone}_group_shap_contribution_data.csv" 
            csv_path_for_zone = os.path.join(input_zone_specific_dir, csv_filename)
            logging.info(f"Attempting to read from primary data source: {csv_path_for_zone}")
        
        # logging.info(f"Attempting to read CSV from: {csv_path_for_zone}") # This line is now conditional above

        # Construct output plot path
        # Output structure: args.output_dir/climate_zone_name/hourly_stacked_bar_{climate_zone_name}.png
        # output_dir_for_zone = os.path.join(args.output_dir, zone) # Already defined above
        # os.makedirs(output_dir_for_zone, exist_ok=True)
        
        # plot_filename = f"hourly_stacked_bar_{zone}.png" 
        # output_plot_path_for_zone = os.path.join(output_dir_for_zone, plot_filename) # Already defined above
        # logging.info(f"Output plot will be saved to: {output_plot_path_for_zone}")


        # Set title for the plot (using LaTeX formatted names if available)
        title_for_zone = get_latex_label(zone)
        title_for_zone = title_for_zone.capitalize() # Ensure first letter is capitalized for the title

        plot_feature_group_from_csv(
            csv_path=csv_path_for_zone,
            output_path=output_plot_path_for_zone,
            title=title_for_zone,
            group_by_column=args.group_by_column,
            base_value_path=args.base_value_path,  # Pass the global base_value_path if provided
            save_data_dump=(not args.read_from_plot_data_dump) # Control saving based on read mode
        )

    logging.info("--- All climate zones processed ---")