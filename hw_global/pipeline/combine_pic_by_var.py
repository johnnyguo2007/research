import os
import argparse
from PIL import Image

def combine_images(var, input_dir, output_dir):
    # Define the filenames for each image
    uhi_diff_file = os.path.join(input_dir, "UHI_diff_global_map_daytime_pcolormesh.png")
    hw_nohw_diff_file = os.path.join(input_dir, f"hw_nohw_diff_{var}_global_map_daytime_pcolormesh.png")
    delta_file = os.path.join(input_dir, f"Delta_{var}_global_map_daytime_pcolormesh.png")
    var_file = os.path.join(input_dir, f"{var}_global_map_daytime_pcolormesh.png")

    # Create a new blank image with the specified dimensions
    combined_image = Image.new("RGB", (11324, 6308), color="white")

    # Open and paste each image if it exists, resizing them to a consistent size
    if os.path.exists(uhi_diff_file):
        uhi_diff_image = Image.open(uhi_diff_file)
        uhi_diff_image = uhi_diff_image.resize((5662, 3154))
        combined_image.paste(uhi_diff_image, (0, 0))

    if os.path.exists(hw_nohw_diff_file):
        hw_nohw_diff_image = Image.open(hw_nohw_diff_file)
        hw_nohw_diff_image = hw_nohw_diff_image.resize((5662, 3154))
        combined_image.paste(hw_nohw_diff_image, (5662, 0))

    if os.path.exists(delta_file):
        delta_image = Image.open(delta_file)
        delta_image = delta_image.resize((5662, 3154))
        combined_image.paste(delta_image, (0, 3154))

    if os.path.exists(var_file):
        var_image = Image.open(var_file)
        var_image = var_image.resize((5662, 3154))
        combined_image.paste(var_image, (5662, 3154))

    # Save the combined image
    output_file = os.path.join(output_dir, f"combined_{var}.png")
    combined_image.save(output_file)

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Combine images for each variable.")
    parser.add_argument("-i", "--input_dir", default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/plots/day",
                        help="Input directory path")
    parser.add_argument("-o", "--output_dir", default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/plots/day/combined_plots",
                        help="Output directory path")
    args = parser.parse_args()

    # Get unique variables from filenames
    variables = set()
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".png"):
            var = filename.split("_")[0]
            variables.add(var)

    # Combine images for each variable
    for var in variables:
        combine_images(var, args.input_dir, args.output_dir)