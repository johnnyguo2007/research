from PIL import Image, ImageDraw, ImageFont

# Load the two images
image1 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_7/Figure_7_stacked_bar_through_hour_Arid.png')
image2 = Image.open('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_7/Figure_7_stacked_bar_through_hour_Tropical.png')

# Detect the width and height of each image
width1, height1 = image1.size
width2, height2 = image2.size

# Calculate the layout dimensions
combined_width = width1 + width2
combined_height = max(height1, height2)

# Create a new blank image for the layout
combined_image = Image.new('RGB', (combined_width, combined_height))

# Paste the images into the layout
combined_image.paste(image1, (0, 0))  # Left
combined_image.paste(image2, (width1, 0))  # Right

# Initialize ImageDraw
draw = ImageDraw.Draw(combined_image)

# Define font (using a default font)
try:
    font = ImageFont.truetype("arial.ttf", 36)
except IOError:
    font = ImageFont.load_default()

# Add text 'a' and 'b'
draw.text((10, 10), "a", font=font, fill="black")
draw.text((width1 + 10, 10), "b", font=font, fill="black")

# Save the combined image
output_path = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/Figure_7/Figure_7_stacked_bar_horizontalCombined.png'
combined_image.save(output_path)
output_path
