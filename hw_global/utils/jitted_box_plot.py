# import matplotlib.pyplot as plt
# import numpy as np

# # Sample data (replace with your actual data)
# self_net = np.random.normal(loc=100, scale=25, size=50)
# care = np.random.normal(loc=90, scale=28, size=50)
# ssai_3d = np.random.normal(loc=5, scale=2, size=50)

# data = [self_net, care, ssai_3d]
# labels = ['Self-Net', 'CARE', 'SSAI-3D']
# colors = [(0.7, 0.7, 0.6), (0.6, 0.5, 0.7), (0.5, 0.7, 0.8)]  # Customize colors

# # Create the plot
# fig, ax = plt.subplots(figsize=(8, 6))

# # Box plot
# bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.4, showfliers=False)

# # Customize box colors
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)

# # Scatter plot (jitter)
# for i, d in enumerate(data):
#     y = d
#     x = np.random.normal(i + 1, 0.08, size=len(y))  # Add jitter
#     ax.scatter(x, y, color=colors[i], alpha=0.5, s=16) # Customize colors and size

# # Set labels and title
# ax.set_ylabel(r'MSE ($\times 10^{-4}$)', fontsize=16)
# ax.set_title('Strands', fontsize=18, loc='left')
# ax.text(-0.1, 1.1, 'f', transform=ax.transAxes, size=20, weight='bold') # Add subplot label
# ax.set_ylim(-5, 155)

# # Show the plot
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data (simulated to match the general look of the chart)
wet_black = np.random.normal(0.4, 0.3, 17)  # 17 values
wet_red = np.random.normal(0.3, 0.25, 17)
wet_blue = np.random.normal(0, 0.2, 17)

intermediate_black = np.random.normal(0.6, 0.4, 101)  # 101 values
intermediate_red = np.random.normal(0.4, 0.3, 101)
intermediate_blue = np.random.normal(0, 0.25, 101)

dry_black = np.random.normal(0.7, 0.35, 15)  # 15 values
dry_red = np.random.normal(0.5, 0.3, 15)
dry_blue = np.random.normal(0.1, 0.25, 15)

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figure size as needed

# Boxplot data - Flattened structure
data = [
    wet_black, wet_red, wet_blue,
    intermediate_black, intermediate_red, intermediate_blue,
    dry_black, dry_red, dry_blue,
]


# Colors for the boxes
colors = ['black', 'red', 'blue']*3
positions = [1,1.25,1.5,2,2.25,2.5,3,3.25,3.5] # Explicit x positions

# Create the box plots
bp = ax.boxplot(
    data,
    positions=positions,
    widths=0.2,
    patch_artist=True,  # Fill with color
    showmeans=True,
    showfliers=True,  # Show outliers
    medianprops={'color': 'black'},
    meanprops={
        'marker': 'x',
        'markeredgecolor': 'black',
        'markerfacecolor': 'black',
    },
    flierprops={'markersize': 3},
)

# Fill boxes with color and set properties
for box, color in zip(bp['boxes'], colors):
    box.set_facecolor('white')  # Set to white
    box.set_edgecolor(color)
    box.set_linewidth(1)

for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1)
    
for whisker,color in zip(bp['whiskers'], colors*2):
    whisker.set_color(color)
    whisker.set_linewidth(1)

for cap,color in zip(bp['caps'], colors*2):
        cap.set_color(color)
        cap.set_linewidth(1)

for flier, color in zip(bp['fliers'], colors*9):
    flier.set_markerfacecolor(color)
    flier.set_markeredgecolor(color)
    
# Customize the plot
ax.set_xticks([1.25, 2.25, 3.25])
ax.set_xticklabels(['Wet', 'Intermediate', 'Dry'], fontsize=14)
ax.set_ylabel('$\u0394T_w$ (°C)', fontsize=14)
ax.set_ylim(-0.8, 1.5)
ax.axhline(0, color='black', linewidth=0.8)  # Add a horizontal line at y=0

# Sample sizes (as shown in the image)
sample_sizes = [17, 101, 15]
for i, size in enumerate(sample_sizes):
    ax.text(
        positions[i*3+1] ,
        -0.9,
        str(size),
        ha='center',
        va='top',
        fontsize=14,
    )

# Add triangles below x-axis labels
triangle_positions = [1.25, 2.25, 3.25]
for pos in triangle_positions:
    ax.plot(pos, -1.1, marker='^', color='black', markersize=8)

# Adjust layout
plt.tight_layout()
plt.show()