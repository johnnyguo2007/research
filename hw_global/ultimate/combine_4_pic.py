from PIL import Image

# Load the four images
image1 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts/910141302561756407/5d6015173a134d6aaadde6cb2d024f37/artifacts/post_process_day_shap_summary_plot.png')
image3 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts/910141302561756407/5d6015173a134d6aaadde6cb2d024f37/artifacts/post_process_day_shap_summary_plot_by_group.png')
image2 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts/571685667182946126/b24ecc84d1f642b48025999d8080fc87/artifacts/post_process_night_shap_summary_plot.png')
image4 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts/571685667182946126/b24ecc84d1f642b48025999d8080fc87/artifacts/post_process_night_shap_summary_plot_by_group.png')

# Detect the width and height of each image
width1, height1 = image1.size
width2, height2 = image2.size
width3, height3 = image3.size
width4, height4 = image4.size

# Calculate the layout dimensions (assume images are the same size)
combined_width = max(width1, width3) * 2
combined_height = max(height1, height2) * 2

# Create a new blank image for the layout
combined_image = Image.new('RGB', (combined_width, combined_height))

# Paste the images into the layout
combined_image.paste(image1, (0, 0))  # Top-left
combined_image.paste(image3, (width1, 0))  # Top-right
combined_image.paste(image2, (0, height1))  # Bottom-left
combined_image.paste(image4, (width1, height1))  # Bottom-right

# Save the combined image
output_path = '/home/jguo/research/hw_global/paper_figure_output/combined_summary_plot.jpg'
combined_image.save(output_path)
output_path
