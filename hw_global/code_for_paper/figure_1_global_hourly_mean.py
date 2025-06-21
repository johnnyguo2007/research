import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
THRESHOLD = 98
# FIGURE_OUTPUT_DIR = '/home/jguo/tmp/output2'
FIGURE_OUTPUT_DIR = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper'
summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'
merged_feather_path = os.path.join(summary_dir, f'updated_local_hour_adjusted_variables_HW{THRESHOLD}.feather')

# Load data
local_hour_adjusted_df = pd.read_feather(merged_feather_path)

# Group the DataFrame by 'local_hour' and calculate the mean and standard deviation of 'UHI_diff'
grouped_df = local_hour_adjusted_df.groupby('local_hour')['UHI_diff'].agg(['mean', 'std'])

# Save grouped_df to a CSV file
grouped_df.to_csv(os.path.join(FIGURE_OUTPUT_DIR, 'Figure_1_global_mean_uhi_by_hour.csv'))

# Plotting UHI_diff Mean and its Standard Deviation
plt.figure(figsize=(10, 6))

# Plot the mean of UHI_diff
plt.plot(grouped_df.index, grouped_df['mean'], marker='o', color='black', linewidth=1, markersize=6)

# Plot the standard deviation as a shaded area around the mean
plt.fill_between(grouped_df.index, grouped_df['mean'] - grouped_df['std'], grouped_df['mean'] + grouped_df['std'], color='grey', alpha=0.15)

plt.xlabel('Local Hour')
plt.ylabel('HW-NHW UHII (°C)')

# Enable horizontal grid lines only
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)  # Horizontal grid ON
plt.grid(False, axis='x')  # Vertical grid OFF

plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, 'Figure_1_global_mean_uhi_by_hour.png'), dpi=600, bbox_inches='tight')
plt.close()
