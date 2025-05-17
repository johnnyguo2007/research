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
from mlflow_tools.plot_util import get_latex_label, replace_cold_with_continental, FEATURE_COLORS

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

        # Get features containing group_name from shap_df
        shap_features_to_select = [col for col in shap_df.columns if group_name in col]

        # Get features containing group_name from feature_values_df and prioritize
        potential_feature_val_features = [col for col in feature_values_df.columns if group_name in col]
        feature_val_features_to_select = potential_feature_val_features
        if len(potential_feature_val_features) > 1:
            double_diff_features = [f for f in potential_feature_val_features if "Double_Differencing" in f]
            if double_diff_features:
                feature_val_features_to_select = double_diff_features

        # Combine selected features for color mapping and further checks
        final_combined_features = sorted(list(set(shap_features_to_select + feature_val_features_to_select)))

        # Check if any features were selected at all
        if not final_combined_features:
            logging.warning(f"No features found for group '{group_name}' in either SHAP or Feature Value data.")
            return

        # Use the predefined FEATURE_COLORS mapping
        color_mapping = FEATURE_COLORS
        # Remove the dynamic color generation
        # palette = sns.color_palette("tab20", n_colors=len(final_combined_features))
        # color_mapping = dict(zip(final_combined_features, palette))

        # Select the identified features from each dataframe
        if not shap_features_to_select:
            logging.warning(f"No features found in SHAP data for group '{group_name}'.")
            # Consider how to handle plotting if one dataframe has no features for the group
            shap_df = pd.DataFrame(index=feature_values_df.index) # Create empty df with same index
        else:
            shap_df = shap_df[shap_features_to_select].copy()

        if not feature_val_features_to_select:
             logging.warning(f"No features found in Feature Value data for group '{group_name}' after prioritization.")
             # Consider how to handle plotting if one dataframe has no features for the group
             feature_values_df = pd.DataFrame(index=shap_df.index) # Create empty df with same index
        else:
             feature_values_df = feature_values_df[feature_val_features_to_select].copy()

        # Final check if both are empty after selection (e.g., if group_name was wrong)
        if shap_df.empty and feature_values_df.empty:
             logging.warning(f"Both SHAP and Feature Value dataframes are empty for group '{group_name}' after selection.")
             return

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

# run command like: /home/jguo/anaconda3/envs/pipJupyter/bin/python /home/jguo/research/hw_global/code_for_paper/figure_9_one_feature_side_by_side.py --group_name Q2M --shap_csv_path "/home/jguo/tmp/output/global/Q2M/shap_contributions_Q2M_global_shap_data.csv" --feature_csv_path "/home/jguo/tmp/output/global/Q2M/shap_and_feature_values_Q2M_global_feature_data.csv"
# To plot all features: /home/jguo/anaconda3/envs/pipJupyter/bin/python /home/jguo/research/hw_global/code_for_paper/figure_9_one_feature_side_by_side.py --shap_csv_path "/home/jguo/tmp/output/global/global_group_shap_contribution_data.csv" --feature_csv_path "/home/jguo/tmp/output/global/shap_and_feature_values_global_feature_data.csv"
if __name__ == "__main__":
    # default_out_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper'
    default_out_dir = '/home/jguo/tmp/output'
    default_input_dir = ("/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/",
                        "summary/mlflow/mlartifacts/",
                        "893793682234305734/67f2e168085e4507b0a79941b74d7eb7/",
                        "artifacts/data_only_24_hourly/"
    )
    climate_zone = "global"
    # default_input_dir = "/home/jguo/tmp/output/global/"
    # climate_zone = "Q2M"
    #path join the default_input_dir
    default_input_dir = os.path.join(*default_input_dir, climate_zone)
    default_shap_csv_path = os.path.join(default_input_dir, "global_group_shap_contribution_data.csv")
    default_feature_csv_path = os.path.join(default_input_dir, "shap_and_feature_values_global_feature_data.csv")
    #print the default_shap_csv_path and default_feature_csv_path   
    print(default_shap_csv_path)
    print(default_feature_csv_path)
    parser = argparse.ArgumentParser(description="Plot side-by-side group plot from CSV data.")
    parser.add_argument("--shap_csv_path", default=default_shap_csv_path, help="Path to the SHAP data CSV file.")
    parser.add_argument("--feature_csv_path", default=default_feature_csv_path, help="Path to the feature data CSV file.")
    parser.add_argument("--output_dir", default=default_out_dir, help="Directory to save the output plot.")
    parser.add_argument("--group_name", default=None, help="Name of the feature group to plot. If not provided, plots all groups defined in FEATURE_COLORS.")
    parser.add_argument("--kg_class", default="global", help="KGMajorClass name.")
    parser.add_argument("--show_total_feature_line", action='store_true', help="Show total feature value line in feature plot.")
    parser.add_argument("--no_total_feature_line", dest='show_total_feature_line', action='store_false', help="Do not show total feature value line.")
    parser.set_defaults(show_total_feature_line=True)


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.group_name:
        # Plot only the specified group
        logging.info(f"Processing specified group: {args.group_name}")
        group_output_dir = os.path.join(args.output_dir, args.kg_class, args.group_name)
        os.makedirs(group_output_dir, exist_ok=True)
        plot_side_by_side_from_csv(
            shap_csv_path=args.shap_csv_path,
            feature_csv_path=args.feature_csv_path,
            output_dir=group_output_dir,
            group_name=args.group_name,
            kg_class=args.kg_class,
            show_total_feature_line=args.show_total_feature_line,
        )
    else:
        # Plot all feature groups defined in FEATURE_COLORS
        logging.info("No specific group provided. Processing all groups from FEATURE_COLORS.")
        for group_name in FEATURE_COLORS.keys():
            logging.info(f"Processing group: {group_name}")
            group_output_dir = os.path.join(args.output_dir, args.kg_class, group_name)
            os.makedirs(group_output_dir, exist_ok=True)
            plot_side_by_side_from_csv(
                shap_csv_path=args.shap_csv_path,
                feature_csv_path=args.feature_csv_path,
                output_dir=group_output_dir,
                group_name=group_name,
                kg_class=args.kg_class,
                show_total_feature_line=args.show_total_feature_line,
            )