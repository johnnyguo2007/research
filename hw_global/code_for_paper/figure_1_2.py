import pandas as pd
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt

THRESHOLD: int = 98

def replace_cold_with_continental(kg_main_group):
    if kg_main_group == 'Cold':
        return 'Continental'
    return kg_main_group


summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'
FIGURE_OUTPUT_DIR = '/home/jguo/tmp/output2'
merged_feather_path = os.path.join(summary_dir, f'updated_local_hour_adjusted_variables_HW{THRESHOLD}.feather')

local_hour_adjusted_df = pd.read_feather(merged_feather_path)

# Group by 'lat', 'lon', and 'local_hour', then calculate the mean for 'UHI_diff'
var_diff_by_localhour = local_hour_adjusted_df.groupby(['lat', 'lon', 'local_hour'])[['UHI_diff']].mean().reset_index().sort_values(by=['lat', 'lon', 'local_hour'])


# Load the Koppen-Geiger climate classification map
ds_koppen_map = xr.open_dataset('/home/jguo/other_projects/1991_2020/koppen_geiger_0p5.nc')

# Load the Koppen-Geiger Legend
kg_legend = pd.read_excel('/home/jguo/research/hw_global/Data/KoppenGeigerLegend.xlsx', engine='openpyxl')

# Convert latitudes and longitudes to numpy arrays
latitudes = ds_koppen_map['lat'].values
longitudes = ds_koppen_map['lon'].values

# Flatten the latitudes, longitudes, and kg_class
lat_flat = np.repeat(latitudes, len(longitudes))
lon_flat = np.tile(longitudes, len(latitudes))
kg_class_flat = ds_koppen_map['kg_class'].values.flatten()

# Filter out the zero kg_class values
non_zero_indices = kg_class_flat > 0
lat_flat_non_zero = lat_flat[non_zero_indices]
lon_flat_non_zero = lon_flat[non_zero_indices]
kg_class_flat_non_zero = kg_class_flat[non_zero_indices]

# Function to find the nearest non-zero Koppen-Geiger class
def find_nearest_non_zero_kg_class(lat, lon):
    distances = np.sqrt((lat_flat_non_zero - lat)**2 + (lon_flat_non_zero - lon)**2)
    nearest_index = np.argmin(distances)
    return kg_class_flat_non_zero[nearest_index]

# Vectorize the function
vec_find_nearest_non_zero_kg_class = np.vectorize(find_nearest_non_zero_kg_class)

# Apply the function to map each grid cell to its nearest Koppen-Geiger class
var_diff_by_localhour['KG_ID'] = vec_find_nearest_non_zero_kg_class(var_diff_by_localhour['lat'].values, var_diff_by_localhour['lon'].values)


def get_kg_color(kg_main_group):
    kg_main_group_colors = {'Tropical': '#0078ff', 'Arid': '#ff9696', 'Temperate': '#96ff96', 'Cold': '#ff00ff'}
    return kg_main_group_colors.get(kg_main_group, '#000000')


# Extract the main climate group
kg_legend['KGMainGroup'] = kg_legend['KGClass'].apply(lambda x: x.split(',')[0].strip())

# Create a mapping from KG class IDs to their main climate groups
kg_main_group_map = dict(zip(kg_legend['ID'], kg_legend['KGMainGroup']))

# Get the unique main group values sorted by their minimum IDs
main_group_min_id = {}
for kg_id, main_group in kg_main_group_map.items():
    if main_group not in main_group_min_id:
        main_group_min_id[main_group] = kg_id
    else:
        main_group_min_id[main_group] = min(main_group_min_id[main_group], kg_id)

sorted_main_groups = sorted(set(kg_main_group_map.values()), key=lambda x: main_group_min_id[x])

# Remove the 'Polar' group
sorted_main_groups = [group for group in sorted_main_groups if group.lower() != 'polar']

# Define Koppen-Geiger color convention
kg_main_group_colors = {
    main_group: get_kg_color(main_group)
    for main_group in ['Tropical', 'Arid', 'Temperate', 'Cold']
}

# Add main group to var_diff_by_localhour
var_diff_by_localhour['KGMainGroup'] = var_diff_by_localhour['KG_ID'].map(kg_main_group_map)

# Calculate average UHI_diff by local_hour for each KG main group
avg_diff_by_hour_and_main_group = var_diff_by_localhour.groupby(['KGMainGroup', 'local_hour'])[['UHI_diff']].agg(['mean', 'std']).reset_index()

# Calculate the global minimum and maximum of the 'mean' UHI_diff
min_uhi_diff = -0.75
max_uhi_diff = 1
# Determine the number of rows needed for subplots
n_rows = (len(sorted_main_groups) + 3) // 4

# Create subplots
fig, axs = plt.subplots(n_rows, 4, figsize=(20, 6*n_rows), squeeze=False)

# Plotting
for i, main_group in enumerate(sorted_main_groups):
    row = i // 4
    col = i % 4

    subset = avg_diff_by_hour_and_main_group[avg_diff_by_hour_and_main_group['KGMainGroup'] == main_group]

    axs[row, col].plot(subset['local_hour'], subset[('UHI_diff', 'mean')], marker='o', color=kg_main_group_colors.get(main_group, 'black'), label='HW-NHW UHI')
    
    axs[row, col].fill_between(
        subset['local_hour'],
        subset[('UHI_diff', 'mean')] - subset[('UHI_diff', 'std')],
        subset[('UHI_diff', 'mean')] + subset[('UHI_diff', 'std')],
        color=kg_main_group_colors.get(main_group, 'gray'),
        alpha=0.2,
        label='±1 Std Dev'
    )
    
    # Create font dictionaries for the title
    title_font = {'size': 14, 'weight': 'normal'}
    main_group_font = {'size': 16, 'weight': 'bold'}

    # Set the title with the main climate group name in bold
    axs[row, col].set_title(f'Climate Zone:', fontdict=title_font)
    axs[row, col].text(0.5, 1.05, replace_cold_with_continental(main_group), fontdict=main_group_font, transform=axs[row, col].transAxes, ha='center', va='bottom')

    axs[row, col].set_xlabel('Local Hour')
    axs[row, col].set_ylabel('Average UHI_diff')
    
    # Enable only horizontal grid lines
    axs[row, col].grid(True, axis='y')
    
    axs[row, col].legend()
    
    # Set consistent y-axis limits across all subplots
    axs[row, col].set_ylim(min_uhi_diff, max_uhi_diff)

# Remove any unused subplots
for i in range(len(sorted_main_groups), n_rows * 4):
    row = i // 4
    col = i % 4
    fig.delaxes(axs[row, col])

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the top spacing
plt.suptitle('Average Hourly HW-NHW UHI by Climate Zone', size=20, weight='bold', y=0.99)  # Add an overall title
plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, 'kg_main_group_uhi_diff.png'), dpi=600, bbox_inches='tight')
plt.close()

# ### Additional Plot: Combined Chart of All Main Groups without Std Dev
plt.figure(figsize=(10, 6))
for main_group in sorted_main_groups:
    subset = avg_diff_by_hour_and_main_group[avg_diff_by_hour_and_main_group['KGMainGroup'] == main_group]
    plt.plot(subset['local_hour'], subset[('UHI_diff', 'mean')], marker='o', color=kg_main_group_colors.get(main_group, 'black'), label=replace_cold_with_continental(main_group))

plt.title('Average Hourly UHI_diff by KG Main Groups')
plt.xlabel('Local Hour')
plt.ylabel('Average UHI_diff')
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
plt.xticks(range(0, 24))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, 'kg_main_group_uhi_diff_combined.png'), dpi=600, bbox_inches='tight')
plt.close()