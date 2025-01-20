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

# Define dummy data for each category and group
wet_black = np.random.normal(loc=0.4, scale=0.3, size=17)  # 17 values
wet_red = np.random.normal(loc=0.3, scale=0.25, size=17)  # 17 values
wet_blue = np.random.normal(loc=0.0, scale=0.2, size=17)  # 17 values

intermediate_black = np.random.normal(loc=0.6, scale=0.4, size=101)  # 101 values
intermediate_red = np.random.normal(loc=0.4, scale=0.3, size=101)  # 101 values
intermediate_blue = np.random.normal(loc=0.0, scale=0.25, size=101)  # 101 values

dry_black = np.random.normal(loc=0.7, scale=0.35, size=15)  # 15 values
dry_red = np.random.normal(loc=0.5, scale=0.3, size=15)  # 15 values
dry_blue = np.random.normal(loc=0.1, scale=0.2, size=15)  # 15 values

# Create the box plot
fig, ax = plt.subplots(figsize=(5, 4))

# Flatten the data structure
data = [wet_black, wet_red, wet_blue,
        intermediate_black, intermediate_red, intermediate_blue,
        dry_black, dry_red, dry_blue]

bplot = ax.boxplot(
    data,
    patch_artist=True,
    showmeans=True,
    showfliers=True,
    positions=[1, 2, 3, 5, 6, 7, 9, 10, 11],
    widths=0.6,
    medianprops=dict(color="black", linewidth=1),
    whiskerprops=dict(color="black", linewidth=1),
    capprops=dict(color="black", linewidth=1),
    flierprops=dict(marker="o", markerfacecolor="black", markersize=2, linestyle="none"),
    meanprops=dict(marker="x", markeredgecolor="black", markersize=6),
    boxprops=dict(color="black", linewidth=1),
)

# Customize box colors
colors = ["black", "red", "blue"] * 3
for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)

# Set y-axis limits and ticks
ax.set_ylim(-0.8, 1.5)
ax.set_yticks([-0.7, 0, 0.7, 1.4])

# Set y-axis label
ax.set_ylabel(r"$\Delta T_w$ (°C)", fontsize=14)

# Add a horizontal line at y=0
ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

# Set x-axis ticks and labels
ax.set_xticks([2, 6, 10])
ax.set_xticklabels(["Wet", "Intermediate", "Dry"], fontsize=14)

# Add sample sizes
sample_sizes = [17, 101, 15]
for i, size in enumerate(sample_sizes):
    ax.text(2 + i * 4, -0.9, size, ha="center", va="top", fontsize=14)

# Add triangles below the x-axis
ax.plot([2, 6, 10], [-1.1, -1.1, -1.1], marker="^", markersize=10, color="black", linestyle="none")

# Remove x-axis ticks
ax.tick_params(axis="x", which="both", bottom=False, top=False)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()