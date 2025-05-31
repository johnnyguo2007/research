import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import logging
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.append('/home/jguo/research/hw_global/ultimate/')
# Assuming plot_side_by_side.py and plot_util.py are in the same directory
from mlflow_tools.plot_side_by_side import create_side_by_side_group_plot, create_combined_plot
from mlflow_tools.plot_util import get_latex_label, replace_cold_with_continental, FEATURE_COLORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the features for the 2x2 composite plot
FEATURES_FOR_2X2_COMPOSITE = ["FIRA", "FSH", "Q2M", "U10"]

def create_2x2_composite_plot(
    kg_class: str,
    base_output_dir_for_kg_class: str,
    features: list[str],
    image_format: str = "png"
):
    """
    Combines four 'combined_shap_and_feature' plots for specified features
    into a single 2x2 composite image for a given kg_class.

    Args:
        kg_class: The climate zone or KGMajorClass (e.g., "Arid").
        base_output_dir_for_kg_class: The directory where individual feature plots
                                      (within their respective subdirectories) are located,
                                      and where the composite plot will be saved.
                                      Example: /home/jguo/tmp/output/Arid
        features: A list of four feature names (group_name) whose plots are to be combined.
                  The order determines their position:
                  features[0]: top-left, features[1]: top-right,
                  features[2]: bottom-left, features[3]: bottom-right.
        image_format: The file extension of the images (default: "png").
    """
    if len(features) != 4:
        logging.error(f"Exactly 4 features are required for a 2x2 composite plot for {kg_class}. Got {len(features)}.")
        return

    loaded_images = []
    image_paths_info = [] # For logging

    for feature_name in features:
        plot_filename = f"combined_shap_and_feature_{feature_name}_{kg_class}.{image_format}"
        image_path = os.path.join(base_output_dir_for_kg_class, feature_name, plot_filename)
        image_paths_info.append(image_path) # Store for logging

        try:
            img = Image.open(image_path)
            loaded_images.append(img)
            logging.debug(f"Successfully loaded image for composite: {image_path}")
        except FileNotFoundError:
            logging.error(f"Image not found for 2x2 composite: {image_path}. Skipping composite plot for {kg_class}.")
            return
        except Exception as e:
            logging.error(f"Error loading image {image_path} for 2x2 composite: {e}. Skipping composite plot for {kg_class}.")
            return

    if len(loaded_images) != 4:
        # This case should ideally be caught by the FileNotFoundError above, but as a safeguard:
        logging.warning(f"Could not load all 4 required images for 2x2 composite plot for {kg_class} (loaded {len(loaded_images)}). Composite plot will not be generated.")
        return

    # Determine target size (max width and height of the loaded images)
    max_width = 0
    max_height = 0
    for img in loaded_images:
        if img.width > max_width: max_width = img.width
        if img.height > max_height: max_height = img.height
    
    if max_width == 0 or max_height == 0:
        logging.error(f"Max width or height for images is zero for {kg_class}. Cannot create composite image.")
        return

    resized_images = [img.resize((max_width, max_height), Image.LANCZOS) for img in loaded_images]

    composite_width = max_width * 2
    composite_height = max_height * 2
    composite_image = Image.new('RGB', (composite_width, composite_height), color='white')

    composite_image.paste(resized_images[0], (0, 0))
    composite_image.paste(resized_images[1], (max_width, 0))
    composite_image.paste(resized_images[2], (0, max_height))
    composite_image.paste(resized_images[3], (max_width, max_height))

    try:
        font_path = "arial.ttf"
        font_size = max(18, int(min(max_width, max_height) * 0.04)) # Reduced font size
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        logging.warning(f"Font '{font_path}' not found for {kg_class} composite plot. Using default font.")
        # Attempt to load a larger default font if possible, otherwise standard default
        try:
            font = ImageFont.load_default(size=max(13, int(min(max_width, max_height) * 0.035))) # Reduced default font size too
        except TypeError: # Older Pillow versions might not support size argument for load_default
            font = ImageFont.load_default()


    draw = ImageDraw.Draw(composite_image)
    plot_labels = [f"({chr(97+i)})" for i in range(len(features))] # Remove feature name from label
    margin = int(min(max_width, max_height) * 0.015) # Reduced margin to bring label closer

    positions = [
        (margin, margin),
        (max_width + margin, margin),
        (margin, max_height + margin),
        (max_width + margin, max_height + margin)
    ]

    for label_text, pos in zip(plot_labels, positions):
        # Basic text shadow by drawing dark then light text (optional, can be removed if problematic)
        # shadow_offset = max(1, font_size // 20)
        # draw.text((pos[0]+shadow_offset, pos[1]+shadow_offset), label_text, font=font, fill=(100,100,100)) # Shadow
        draw.text(pos, label_text, font=font, fill=(0,0,0)) # Main text

    output_filename = f"{kg_class}_composite_2x2_{'_'.join(features)}.{image_format}"
    output_path = os.path.join(base_output_dir_for_kg_class, output_filename)
    print(f"Saving composite 2x2 plot to {output_path}")
    
    try:
        composite_image.save(output_path)
        logging.info(f"Composite 2x2 plot for {kg_class} (features: {', '.join(features)}) saved to {output_path}")
        logging.info(f"Images used for composite: {', '.join(image_paths_info)}")
    except Exception as e:
        logging.error(f"Error saving composite image {output_path}: {e}")


def create_feature_climate_composite_plot(
    feature_to_composite: str,
    base_overall_output_dir: str, # e.g., /home/jguo/tmp/output
    climate_zones: list[str],
    image_format: str = "png"
):
    """
    Combines four 'combined_shap_and_feature' plots for a specified feature,
    each from a different climate zone, into a single 2x2 composite image.

    Args:
        feature_to_composite: The feature name (e.g., "FIRA") for which plots are combined.
        base_overall_output_dir: The main output directory where climate zone subdirectories
                                 (e.g., Arid, Continental) are located.
                                 Example: /home/jguo/tmp/output
        climate_zones: A list of four climate zone names (kg_class).
                       The order determines their position:
                       climate_zones[0]: top-left, climate_zones[1]: top-right,
                       climate_zones[2]: bottom-left, climate_zones[3]: bottom-right.
        image_format: The file extension of the images (default: "png").
    """
    if len(climate_zones) != 4:
        logging.error(f"Exactly 4 climate zones are required for a 2x2 composite plot for feature {feature_to_composite}. Got {len(climate_zones)}.")
        return

    loaded_images = []
    image_paths_info = [] # For logging

    for zone_name in climate_zones:
        # Path: {base_overall_output_dir}/{zone_name}/{feature_to_composite}/combined_shap_and_feature_{feature_to_composite}_{zone_name}.{image_format}
        plot_filename = f"combined_shap_and_feature_{feature_to_composite}_{zone_name}.{image_format}"
        # The individual plot is in a subdirectory named after the feature, within the climate zone's directory
        image_path = os.path.join(base_overall_output_dir, zone_name, feature_to_composite, plot_filename)
        image_paths_info.append(image_path)

        try:
            img = Image.open(image_path)
            loaded_images.append(img)
            logging.debug(f"Successfully loaded image for feature-climate composite: {image_path}")
        except FileNotFoundError:
            logging.error(f"Image not found for feature-climate composite: {image_path}. Skipping composite for feature {feature_to_composite}.")
            return
        except Exception as e:
            logging.error(f"Error loading image {image_path} for feature-climate composite: {e}. Skipping composite for feature {feature_to_composite}.")
            return

    if len(loaded_images) != 4:
        logging.warning(f"Could not load all 4 required images for feature-climate composite plot for {feature_to_composite} (loaded {len(loaded_images)}). Composite plot will not be generated.")
        return

    max_width = 0
    max_height = 0
    for img in loaded_images:
        if img.width > max_width: max_width = img.width
        if img.height > max_height: max_height = img.height

    if max_width == 0 or max_height == 0:
        logging.error(f"Max width or height for images is zero for feature {feature_to_composite}. Cannot create composite image.")
        return

    resized_images = [img.resize((max_width, max_height), Image.LANCZOS) for img in loaded_images]

    composite_width = max_width * 2
    composite_height = max_height * 2
    composite_image = Image.new('RGB', (composite_width, composite_height), color='white')

    composite_image.paste(resized_images[0], (0, 0))
    composite_image.paste(resized_images[1], (max_width, 0))
    composite_image.paste(resized_images[2], (0, max_height))
    composite_image.paste(resized_images[3], (max_width, max_height))

    try:
        font_path = "arial.ttf"
        font_size = max(16, int(min(max_width, max_height) * 0.04))
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        logging.warning(f"Font '{font_path}' not found for {feature_to_composite} feature-climate composite plot. Using default font.")
        try:
            font = ImageFont.load_default(size=max(12, int(min(max_width, max_height) * 0.035)))
        except TypeError:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(composite_image)
    # Labels are the climate zones
    plot_labels = [f"({chr(97+i)}) {get_latex_label(zone)}" for i, zone in enumerate(climate_zones)] # Use get_latex_label for consistency if needed
    margin = int(min(max_width, max_height) * 0.025)

    positions = [
        (margin, margin),
        (max_width + margin, margin),
        (margin, max_height + margin),
        (max_width + margin, max_height + margin)
    ]

    for label_text, pos in zip(plot_labels, positions):
        draw.text(pos, label_text, font=font, fill=(0,0,0))

    # Output filename: {base_overall_output_dir}/{feature_to_composite}_climate_zones_composite_2x2.{image_format}
    output_filename = f"{feature_to_composite}_climate_zones_composite_2x2.{image_format}"
    output_path = os.path.join(base_overall_output_dir, output_filename) # Saved in the base overall output directory
    print(f"Saving feature-climate composite 2x2 plot to {output_path}")

    try:
        composite_image.save(output_path)
        logging.info(f"Feature-climate composite 2x2 plot for {feature_to_composite} (zones: {', '.join(climate_zones)}) saved to {output_path}")
        logging.info(f"Images used for composite: {', '.join(image_paths_info)}")
    except Exception as e:
        logging.error(f"Error saving feature-climate composite image {output_path}: {e}")


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

        # Add call to the new combined plot function
        create_combined_plot(
            shap_df=shap_df,
            feature_values_df=feature_values_df,
            group_name=group_name,
            output_dir=output_dir, # This should be the kg_class_output_dir, group-specific dir is handled inside
            kg_class=kg_class,
            color_mapping=color_mapping,
            show_total_feature_line=show_total_feature_line,
        )
        logging.info(f"Combined plot created and saved in '{output_dir}'.")

    except FileNotFoundError:
        logging.error("CSV file not found. Please check the file paths.")
    except Exception as e:
        logging.error(f"An error occurred while plotting: {e}")

# run command like: /home/jguo/anaconda3/envs/pipJupyter/bin/python /home/jguo/research/hw_global/code_for_paper/figure_9_one_feature_side_by_side.py --group_name Q2M --shap_csv_path "/home/jguo/tmp/output/global/Q2M/shap_contributions_Q2M_global_shap_data.csv" --feature_csv_path "/home/jguo/tmp/output/global/Q2M/shap_and_feature_values_Q2M_global_feature_data.csv"
# To plot all features: /home/jguo/anaconda3/envs/pipJupyter/bin/python /home/jguo/research/hw_global/code_for_paper/figure_9_one_feature_side_by_side.py --shap_csv_path "/home/jguo/tmp/output/global/global_group_shap_contribution_data.csv" --feature_csv_path "/home/jguo/tmp/output/global/shap_and_feature_values_global_feature_data.csv"
if __name__ == "__main__":
    default_out_dir = '/home/jguo/tmp/output'
    # Base input directory tuple, before appending the climate zone
    default_input_dir_base_tuple = ("/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/",
                                    "summary/mlflow/mlartifacts/",
                                    "893793682234305734/67f2e168085e4507b0a79941b74d7eb7/",
                                    "artifacts/data_only_24_hourly/")

    parser = argparse.ArgumentParser(description="Plot side-by-side group plot from CSV data for multiple climate zones.")
    parser.add_argument("--shap_csv_path", default=None, help="Path to the SHAP data CSV file. If not provided, it's constructed based on the climate zone.")
    parser.add_argument("--feature_csv_path", default=None, help="Path to the feature data CSV file. If not provided, it's constructed based on the climate zone.")
    parser.add_argument("--output_dir", default=default_out_dir, help="Base directory to save the output plots.")
    parser.add_argument("--group_name", default=None, help="Name of the feature group to plot. If not provided, plots all groups defined in FEATURE_COLORS.")
    parser.add_argument("--kg_class", default=None, help="KGMajorClass name. If not provided, it's inferred from the climate zone during looping.")
    parser.add_argument("--show_total_feature_line", action='store_true', help="Show total feature value line in feature plot.")
    parser.add_argument("--no_total_feature_line", dest='show_total_feature_line', action='store_false', help="Do not show total feature value line.")
    parser.set_defaults(show_total_feature_line=True)

    args = parser.parse_args()

    climate_zones_to_process = ["Arid", "Continental", "global", "Temperate", "Tropical"]
    # Test with a subset for faster iteration if needed
    # climate_zones_to_process = ["Arid"] 

    for current_climate_zone in climate_zones_to_process:
        logging.info(f"--- Processing climate zone: {current_climate_zone} ---")

        current_input_dir_for_zone = os.path.join(*default_input_dir_base_tuple, current_climate_zone)

        # Determine SHAP and Feature CSV paths for the current zone
        # If user provided specific paths via CLI for args.shap_csv_path or args.feature_csv_path,
        # those would override this per-zone logic. However, the intent of this modification
        # is to loop and generate paths per zone. For simplicity, we assume CLI paths are not
        # intended to override the loop's path generation, or this script is run without them
        # when looping is desired.
        shap_csv_path_for_zone = args.shap_csv_path if args.shap_csv_path else os.path.join(current_input_dir_for_zone, f"{current_climate_zone}_group_shap_contribution_data.csv")
        feature_csv_path_for_zone = args.feature_csv_path if args.feature_csv_path else os.path.join(current_input_dir_for_zone, f"shap_and_feature_values_{current_climate_zone}_feature_data.csv")

        # The KGMajorClass for this iteration is the current climate zone
        # If args.kg_class is specified, it might be intended as a global override,
        # but typically it should align with the climate zone for output organization.
        kg_class_for_zone = args.kg_class if args.kg_class else current_climate_zone

        # Output directory for the current climate_zone/kg_class
        output_dir_for_kg_class = os.path.join(args.output_dir, kg_class_for_zone)
        os.makedirs(output_dir_for_kg_class, exist_ok=True)

        logging.info(f"Using SHAP CSV path: {shap_csv_path_for_zone}")
        logging.info(f"Using Feature CSV path: {feature_csv_path_for_zone}")
        logging.info(f"Output directory for {kg_class_for_zone}: {output_dir_for_kg_class}")

        if args.group_name:
            logging.info(f"Processing specified group: {args.group_name} for climate zone: {current_climate_zone}")
            plot_side_by_side_from_csv(
                shap_csv_path=shap_csv_path_for_zone,
                feature_csv_path=feature_csv_path_for_zone,
                output_dir=output_dir_for_kg_class,
                group_name=args.group_name,
                kg_class=kg_class_for_zone,
                show_total_feature_line=args.show_total_feature_line,
            )
        else:
            logging.info(f"No specific group provided. Processing all groups from FEATURE_COLORS for climate zone: {current_climate_zone}.")
            if not FEATURE_COLORS:
                logging.warning(f"FEATURE_COLORS is empty. Cannot process all groups for climate zone: {current_climate_zone}.")
                continue # Skip to the next climate zone

            for group_name_key in FEATURE_COLORS.keys():
                logging.info(f"Processing group: {group_name_key} for climate zone: {current_climate_zone}")
                plot_side_by_side_from_csv(
                    shap_csv_path=shap_csv_path_for_zone,
                    feature_csv_path=feature_csv_path_for_zone,
                    output_dir=output_dir_for_kg_class,
                    group_name=group_name_key,
                    kg_class=kg_class_for_zone,
                    show_total_feature_line=args.show_total_feature_line,
                )
        
        # After processing all groups (or the specified group) for the current climate zone,
        # attempt to create the 2x2 composite plot.
        logging.info(f"Attempting to create 2x2 composite plot for {kg_class_for_zone} using features: {FEATURES_FOR_2X2_COMPOSITE}")
        create_2x2_composite_plot(
            kg_class=kg_class_for_zone,
            base_output_dir_for_kg_class=output_dir_for_kg_class, # This is the directory like /home/jguo/tmp/output/Arid
            features=FEATURES_FOR_2X2_COMPOSITE,
            image_format="png" # Ensure this matches the output of create_combined_plot
        )

    logging.info("--- All climate zones processed ---")

    # After processing all climate zones and their individual plots/composites:
    logging.info("--- Starting generation of feature-climate composite plots ---")
    
    CLIMATE_ZONES_FOR_FEATURE_COMPOSITE = ["Arid", "Continental", "Temperate", "Tropical"]
    
    processed_zones_set = set(climate_zones_to_process)
    required_zones_for_new_composite_set = set(CLIMATE_ZONES_FOR_FEATURE_COMPOSITE)
    
    if not required_zones_for_new_composite_set.issubset(processed_zones_set):
        missing_zones = list(required_zones_for_new_composite_set - processed_zones_set)
        logging.warning(f"Cannot create feature-climate composites. The following climate zones, required for these composites, were not in the processing list: {missing_zones}. Please ensure they are processed first.")
    else:
        for feature_name_for_fc_composite in FEATURES_FOR_2X2_COMPOSITE:
            logging.info(f"Attempting to create feature-climate 2x2 composite plot for feature: {feature_name_for_fc_composite} using climate zones: {CLIMATE_ZONES_FOR_FEATURE_COMPOSITE}")
            create_feature_climate_composite_plot(
                feature_to_composite=feature_name_for_fc_composite,
                base_overall_output_dir=args.output_dir, 
                climate_zones=CLIMATE_ZONES_FOR_FEATURE_COMPOSITE,
                image_format="png" 
            )
            
    logging.info("--- All processing finished ---")