import os
import argparse
from PIL import Image

def extract_variable(filename):
    patterns = ["UHI_diff_", "Double_Differencing_", "hw_nohw_diff_", "Delta_"]
    for pattern in patterns:
        if filename.startswith(pattern):
            return filename.replace(pattern, "").split("_")[0]
    return filename.split("_")[0]

def combine_images(var, input_dir, output_dir):
    print(f"Combining images for variable: {var}")
    
    # Define the filenames for each image
    image_files = [
        os.path.join(input_dir, "UHI_diff_global_map_daytime_pcolormesh.png"),
        os.path.join(input_dir, f"Double_Differencing_{var}_global_map_daytime_pcolormesh.png"),
        os.path.join(input_dir, f"hw_nohw_diff_{var}_global_map_daytime_pcolormesh.png"),
        os.path.join(input_dir, f"Delta_{var}_global_map_daytime_pcolormesh.png"),
        os.path.join(input_dir, f"{var}_global_map_daytime_pcolormesh.png")
    ]

    # Find the first existing image and get its dimensions
    for file in image_files:
        if os.path.exists(file):
            first_image = Image.open(file)
            width, height = first_image.size
            break
    else:
        print(f"No images found for variable: {var}. Skipping.")
        return

    # Create a new blank image with the adjusted dimensions
    combined_image = Image.new("RGB", (width * 2, height * 3), color="white")

    # Define the positions for each image in the combined image
    positions = [
        (0, 0), (width, 0),
        (0, height), (width, height),
        (0, height * 2)
    ]

    # Open and paste each image if it exists, resizing them to a consistent size
    for file, position in zip(image_files, positions):
        if os.path.exists(file):
            print(f"  Adding {os.path.basename(file)}")
            image = Image.open(file)
            image = image.resize((width, height))
            combined_image.paste(image, position)

    # Save the combined image
    output_file = os.path.join(output_dir, f"combined_{var}.png")
    combined_image.save(output_file)
    print(f"Combined image saved as: {output_file}\n")

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Combine images for each variable.")
    parser.add_argument("-i", "--input_dir", default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/plots/day",
                        help="Input directory path")
    parser.add_argument("-o", "--output_dir", default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/plots/day/combined_plots",
                        help="Output directory path")
    args = parser.parse_args()

    print("Starting image combination process...")
    
    # Get unique variables from filenames
    variables = set()
    print("Extracting variables from filenames...")
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".png"):
            var = extract_variable(filename)
            variables.add(var)

    print(f"Found {len(variables)} unique variables: {', '.join(variables)}\n")

    # Combine images for each variable
    for var in variables:
        combine_images(var, args.input_dir, args.output_dir)

    print("Image combination process completed.")