"""
This script combines day and night feature analysis plots side-by-side into three summary figures.
It is designed to work with the outputs of other plotting scripts, taking pairs of images
(one for 'daytime', one for 'nighttime') and merging them into a single comparative image.

The script generates the following three output files:
1.  'feature_importance_day_night.png': Combines the global feature importance plots for day and night.
2.  'feature_summary_day_night.png': Combines the global feature summary plots (beeswarm) for day and night.
3.  'group_feature_summary_day_night.png': Combines the global grouped feature summary plots for day and night.

The input images are expected to be in a directory structure created by a preceding analysis step,
and the final combined images are saved to a dedicated 'Figure_3' directory for the paper.
"""

import os
from PIL import Image, ImageDraw, ImageFont

def combine_images_side_by_side(image_path1, image_path2, output_path, label1="(a)", label2="(b)"):
    """
    Opens two images, resizes them to a consistent height, combines them side-by-side,
    adds labels, and saves the resulting image.

    Args:
        image_path1 (str): Path to the first image (left).
        image_path2 (str): Path to the second image (right).
        output_path (str): Path to save the combined image.
        label1 (str): Label for the first image.
        label2 (str): Label for the second image.
    """
    try:
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)
    except FileNotFoundError as e:
        print(f"Error: Could not open image file. {e}")
        return

    # Standardize height, adjust width proportionally
    base_height = max(img1.height, img2.height)
    img1_width = int(img1.width * (base_height / img1.height))
    img2_width = int(img2.width * (base_height / img2.height))

    img1 = img1.resize((img1_width, base_height), Image.LANCZOS)
    img2 = img2.resize((img2_width, base_height), Image.LANCZOS)

    # Create a new image to hold the combined images
    combined_width = img1.width + img2.width
    combined_image = Image.new('RGB', (combined_width, base_height))

    # Paste the images
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (img1.width, 0))

    # Add labels
    try:
        # Scale font size based on image height
        font_size = max(16, min(48, base_height // 30))
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(combined_image)
    # Add simple labels like (a), (b)
    draw.text((10, 10), label1, (0, 0, 0), font=font)
    draw.text((img1.width + 10, 10), label2, (0, 0, 0), font=font)

    # Save the combined image
    combined_image.save(output_path)
    print(f"Saved combined image to: {output_path}")


def main():
    """
    Main function to define paths and orchestrate the image combination process.
    """
    default_input_dir_base_tuple = ("/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/",
                                "summary/mlflow/mlartifacts/",
                                "893793682234305734/67f2e168085e4507b0a79941b74d7eb7/",
                                "artifacts/24_hourly_plot/")
    # Directory where the source images are located
    base_dir = os.path.join(*default_input_dir_base_tuple)
    
    # Directory where the final combined figures will be saved
    output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_3'
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Combine Feature Importance Plots ---
    day_importance_path = os.path.join(base_dir, 'daytime_feature_summary_plots/summary_plots/feature_importance_plot_Global.png')
    night_importance_path = os.path.join(base_dir, 'nighttime_feature_summary_plots/summary_plots/feature_importance_plot_Global.png')
    output_importance_path = os.path.join(output_dir, 'feature_importance_day_night.png')
    
    combine_images_side_by_side(
        day_importance_path,
        night_importance_path,
        output_importance_path,
        label1="(a) Day",
        label2="(b) Night"
    )

    # --- 2. Combine Feature Summary (Beeswarm) Plots ---
    day_summary_path = os.path.join(base_dir, 'daytime_feature_summary_plots/summary_plots/feature_summary_plot_Global.png')
    night_summary_path = os.path.join(base_dir, 'nighttime_feature_summary_plots/summary_plots/feature_summary_plot_Global.png')
    output_summary_path = os.path.join(output_dir, 'feature_summary_day_night.png')

    combine_images_side_by_side(
        day_summary_path,
        night_summary_path,
        output_summary_path,
        label1="(a) Day",
        label2="(b) Night"
    )

    # --- 3. Combine Grouped Feature Summary Plots ---
    day_group_summary_path = os.path.join(base_dir, 'daytime_group_summary_plots/summary_plots/group_summary_plot_Global.png')
    night_group_summary_path = os.path.join(base_dir, 'nighttime_group_summary_plots/summary_plots/group_summary_plot_Global.png')
    output_group_summary_path = os.path.join(output_dir, 'group_summary_day_night.png')

    combine_images_side_by_side(
        day_group_summary_path,
        night_group_summary_path,
        output_group_summary_path,
        label1="(a) Day",
        label2="(b) Night"
    )

    # --- 4. Combine Group Importance Plots ---
    day_group_importance_path = os.path.join(base_dir, 'daytime_group_summary_plots/summary_plots/feature_importance_plot_Global.png')
    night_group_importance_path = os.path.join(base_dir, 'nighttime_group_summary_plots/summary_plots/feature_importance_plot_Global.png')
    output_group_importance_path = os.path.join(output_dir, 'group_importance_day_night.png')

    combine_images_side_by_side(
        day_group_importance_path,
        night_group_importance_path,
        output_group_importance_path,
        label1="(a) Day",
        label2="(b) Night"
    )

if __name__ == "__main__":
    main()
