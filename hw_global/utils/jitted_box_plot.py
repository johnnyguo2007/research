import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual data)
self_net = np.random.normal(loc=100, scale=25, size=50)
care = np.random.normal(loc=90, scale=28, size=50)
ssai_3d = np.random.normal(loc=5, scale=2, size=50)

data = [self_net, care, ssai_3d]
labels = ['Self-Net', 'CARE', 'SSAI-3D']
colors = [(0.7, 0.7, 0.6), (0.6, 0.5, 0.7), (0.5, 0.7, 0.8)]  # Customize colors

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Box plot
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.4, showfliers=False)

# Customize box colors
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Scatter plot (jitter)
for i, d in enumerate(data):
    y = d
    x = np.random.normal(i + 1, 0.08, size=len(y))  # Add jitter
    ax.scatter(x, y, color=colors[i], alpha=0.5, s=16) # Customize colors and size

# Set labels and title
ax.set_ylabel(r'MSE ($\times 10^{-4}$)', fontsize=16)
ax.set_title('Strands', fontsize=18, loc='left')
ax.text(-0.1, 1.1, 'f', transform=ax.transAxes, size=20, weight='bold') # Add subplot label
ax.set_ylim(-5, 155)

# Show the plot
plt.tight_layout()
plt.show()