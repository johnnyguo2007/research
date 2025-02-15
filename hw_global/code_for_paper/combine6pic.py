from PIL import Image, ImageDraw, ImageFont

# Load the six images
image1 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_3_a_feature_summary_plot_Global.png')
image2 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_3_b_group_summary_plot_Global.png')
image4 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_3_c_feature_summary_plot_Global.png')
image5 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_3_d_group_summary_plot_Global.png')
image3 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_3_e_feature_importance_plot_Global.png')
image6 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_3_f_feature_importance_plot_Global.png')


# Define the target size for scaling
max_width = max(image1.width, image2.width, image3.width, image4.width, image5.width, image6.width)
max_height = max(image1.height, image2.height, image3.height, image4.height, image5.height, image6.height)

# Resize images to the target size
image1 = image1.resize((max_width, max_height), Image.LANCZOS)
image2 = image2.resize((max_width, max_height), Image.LANCZOS)
image3 = image3.resize((max_width, max_height), Image.LANCZOS)
image4 = image4.resize((max_width, max_height), Image.LANCZOS)
image5 = image5.resize((max_width, max_height), Image.LANCZOS)
image6 = image6.resize((max_width, max_height), Image.LANCZOS)

# Update width and height variables
width, height = max_width, max_height

# Calculate the layout dimensions
combined_width = width * 3
combined_height = height * 2

# Create a new blank image for the layout
combined_image = Image.new('RGB', (combined_width, combined_height))

# Paste the images into the layout
combined_image.paste(image1, (0, 0))  # Top-left
combined_image.paste(image2, (width, 0))  # Top-center
combined_image.paste(image3, (width * 2, 0))  # Top-right
combined_image.paste(image4, (0, height))  # Bottom-left
combined_image.paste(image5, (width, height))  # Bottom-center
combined_image.paste(image6, (width * 2, height))  # Bottom-right

# Load a font with increased size
try:
    font = ImageFont.truetype("arial.ttf", 48)
except IOError:
    font = ImageFont.load_default()

# Create a draw object
draw = ImageDraw.Draw(combined_image)

# Define labels and positions with an offset for visibility
labels = ['a', 'b', 'e', 'c', 'd', 'f']
# Adding a small offset to the positions to ensure labels are visible
positions = [(10, 10), (width + 10, 10), (width * 2 + 10, 10), (10, height + 10), (width + 10, height + 10), (width * 2 + 10, height + 10)]

# Draw labels on the images with black color for better contrast
for label, position in zip(labels, positions):
    draw.text(position, label, (0, 0, 0), font=font)

# Save the combined image
output_path = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_3_combined_6_images.jpg'
combined_image.save(output_path)
output_path
