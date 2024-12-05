import pandas as pd
import numpy as np
import xarray as xr
import os
THRESHOLD: int = 98

# #  3: Load Local Hour Adjusted Variables

summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'

FIGURE_OUTPUT_DIR = '/home/jguo/research/hw_global/paper_figure_output'

# merged_feather_path = os.path.join(summary_dir, 'local_hour_adjusted_variables.feather')
merged_feather_path = os.path.join(summary_dir, f'updated_local_hour_adjusted_variables_HW{THRESHOLD}.feather')

local_hour_adjusted_df = pd.read_feather(merged_feather_path)
local_hour_adjusted_df.info()

# ##  3.1: Calculate the Number of Unique Locations 

len(local_hour_adjusted_df['location_ID'].unique()) 

# ##  3.2: Compute Average Differences Based on Local Hour

# os.environ["PROJ_LIB"] = "/home/jguo/anaconda3/envs/I2000/share/proj"
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Group by 'lat', 'lon', and 'local_hour', then calculate the mean for 'UHI_diff' and 'UWBI_diff'
# Ensure the grouped data is sorted by 'lat', 'lon', and 'local_hour' before resetting the index
var_diff_by_localhour = local_hour_adjusted_df.groupby(['lat', 'lon', 'local_hour'])[['UHI_diff', 'UWBI_diff']].mean().reset_index().sort_values(by=['lat', 'lon', 'local_hour'])

var_diff_by_localhour

var_diff_by_localhour.info()

# # 4: Plotting and Data Exploration

# ## 4.1: Visualize UHI_diff and UWBI_diff by Local Hour

# Plot control switches
PLOT_GLOBAL_MEAN = True
PLOT_KG_CLASS = False
PLOT_KG_MAIN_GROUP = True

if PLOT_GLOBAL_MEAN:
    # Group the DataFrame by 'local_hour' and calculate the mean and standard deviation of 'UHI_diff'
    grouped_df = local_hour_adjusted_df.groupby('local_hour')['UHI_diff'].agg(['mean', 'std'])
    
    # Plotting UHI_diff Mean and its Standard Deviation
    plt.figure(figsize=(10, 6))
    
    # Plot the mean of UHI_diff
    plt.plot(grouped_df.index, grouped_df['mean'], marker='o', label='UHI_diff Mean')
    
    # Plot the standard deviation as a shaded area around the mean
    plt.fill_between(grouped_df.index, grouped_df['mean'] - grouped_df['std'], grouped_df['mean'] + grouped_df['std'], color='cornflowerblue', alpha=0.1, label='UHI_diff ±1 Std Dev')
    
    plt.title('Global Mean UHI Difference by Local Hour')
    plt.xlabel('Local Hour')
    plt.ylabel('Mean UHI Difference')
    
    # Enable horizontal grid lines only
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)  # Horizontal grid ON
    plt.grid(False, axis='x')  # Vertical grid OFF
    
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, 'global_mean_uhi_by_hour.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ##  4.2: Koppen Geiger Climate Analysis

# ###  4.2.1: Load the Koppen Geiger Map and Legend

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load the Koppen-Geiger climate classification map from a NetCDF file
ds_koppen_map = xr.open_dataset('/home/jguo/other_projects/1991_2020/koppen_geiger_0p5.nc')
#ds_koppen_map.kg_class.min()

# Load the Koppen-Geiger Legend from an Excel file for mapping class IDs to descriptions
kg_legend = pd.read_excel('/home/jguo/research/hw_global/Data/KoppenGeigerLegend.xlsx', engine='openpyxl')
kg_legend

# ###  4.2.2: Assign the Nearest Koppen Geiger Class to Each Grid Cell

# Convert latitudes and longitudes from the NetCDF dataset to numpy arrays
latitudes = ds_koppen_map['lat'].values
longitudes = ds_koppen_map['lon'].values

# Flatten the latitudes, longitudes, and kg_class for easier manipulation
lat_flat = np.repeat(latitudes, len(longitudes))
lon_flat = np.tile(longitudes, len(latitudes))
kg_class_flat = ds_koppen_map['kg_class'].values.flatten()

# Filter out the zero kg_class values
non_zero_indices = kg_class_flat > 0
lat_flat_non_zero = lat_flat[non_zero_indices]
lon_flat_non_zero = lon_flat[non_zero_indices]
kg_class_flat_non_zero = kg_class_flat[non_zero_indices]

# Function to find the nearest non-zero Koppen-Geiger class for a given latitude and longitude
def find_nearest_non_zero_kg_class(lat, lon):
    distances = np.sqrt((lat_flat_non_zero - lat)**2 + (lon_flat_non_zero - lon)**2)
    nearest_index = np.argmin(distances)
    return kg_class_flat_non_zero[nearest_index]

# Vectorize the function to apply it efficiently to arrays
vec_find_nearest_non_zero_kg_class = np.vectorize(find_nearest_non_zero_kg_class)

# Apply the function to map each grid cell to its nearest Koppen-Geiger class
var_diff_by_localhour['KG_ID'] = vec_find_nearest_non_zero_kg_class(var_diff_by_localhour['lat'].values, var_diff_by_localhour['lon'].values)
var_diff_by_localhour.head()

# ###  4.2.3: Plot Average UHI_diff by Local Hour for Each Koppen Geiger Class
if PLOT_KG_CLASS:
    # Calculate average UHI_diff by local_hour for each Koppen-Geiger class
    avg_uhi_by_hour_and_kg = var_diff_by_localhour.groupby(['KG_ID', 'local_hour'])['UHI_diff'].mean().reset_index()
    
    # Create a mapping from KG class IDs to their descriptive names using the legend
    kg_map = dict(zip(kg_legend['ID'], kg_legend['KGClass']))
    
    # Plotting
    import textwrap
    
    # Define the number of graphs you want in each row
    graphs_per_row = 4  # You can change this number to your preference
    
    # Find the global minimum and maximum UHI_diff values for consistent y-axis limits
    global_min_uhi = avg_uhi_by_hour_and_kg['UHI_diff'].min()
    global_max_uhi = avg_uhi_by_hour_and_kg['UHI_diff'].max()
    
    # Unique KG IDs
    unique_kg_ids = avg_uhi_by_hour_and_kg['KG_ID'].unique()
    
    # Number of KG IDs
    n_kg_ids = len(unique_kg_ids)
    
    # Determine the number of rows for subplots based on the number of Koppen-Geiger classes
    n_rows = (n_kg_ids + graphs_per_row - 1) // graphs_per_row  # Ensures rounding up
    
    # Generate plots for each Koppen-Geiger class showing the average UHI_diff by local hour
    for i, kg_id in enumerate(unique_kg_ids):
        # Create a new figure at the start and after every 'graphs_per_row' plots
        if i % graphs_per_row == 0:
            fig = plt.figure(figsize=(5 * graphs_per_row, 5 * n_rows))  # Adjust figure size as needed
        # Select the subplot position
        plt.subplot(n_rows, graphs_per_row, i % graphs_per_row + 1)
    
        # Extract the subset of data for the current KG ID
        subset = avg_uhi_by_hour_and_kg[avg_uhi_by_hour_and_kg['KG_ID'] == kg_id]
    
        # Plot the average UHI_diff
        plt.plot(subset['local_hour'], subset['UHI_diff'], marker='o')
    
        # Wrap the title text for better readability
        title_text = f'KG Class {kg_id}: {kg_map.get(kg_id, "Unknown")} - Average Hourly UHI_diff'
        wrapped_title = textwrap.fill(title_text, width=40)  # Adjust 'width' as needed
    
        plt.title(wrapped_title)
        plt.xlabel('Local Hour')
        plt.ylabel('Average UHI_diff')
        plt.grid(True)
    
        # Set consistent y-axis limits across all plots
        plt.ylim(global_min_uhi, global_max_uhi)
    
        # Display the figure after every 'graphs_per_row' plots or on the last plot
        if (i % graphs_per_row == graphs_per_row - 1) or (i == n_kg_ids - 1):
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, f'kg_class_uhi_diff_group_{i//graphs_per_row + 1}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()

# ###  4.2.4: Main Group Analysis - Aggregate Data by Main Koppen Geiger Groups
if PLOT_KG_MAIN_GROUP:
    # The kg_legend data frame has a KGClass column, which has values for KG classification main group and subgroup, separated by a comma.
    # Extract the main climate group from the detailed Koppen-Geiger class descriptions
    kg_legend['KGMainGroup'] = kg_legend['KGClass'].apply(lambda x: x.split(',')[0].strip())
    
    # Create a mapping from KG class IDs to their main climate groups
    kg_main_group_map = dict(zip(kg_legend['ID'], kg_legend['KGMainGroup']))
    kg_main_group_map
    
    set(kg_main_group_map.values())
    
    # Extract main group from KGClass
    kg_legend['KGMainGroup'] = kg_legend['KGClass'].apply(lambda x: x.split(',')[0].strip())
    
    # Map KG IDs to their main groups
    kg_main_group_map = dict(zip(kg_legend['ID'], kg_legend['KGMainGroup']))
    
    # Create a dictionary to map main group values to their minimum IDs
    main_group_min_id = {}
    for kg_id, main_group in kg_main_group_map.items():
        if main_group not in main_group_min_id:
            main_group_min_id[main_group] = kg_id
        else:
            main_group_min_id[main_group] = min(main_group_min_id[main_group], kg_id)
    
    # Get the unique main group values sorted by their minimum IDs
    sorted_main_groups = sorted(set(kg_main_group_map.values()), key=lambda x: main_group_min_id[x])
    
    # Add main group to var_diff_by_localhour
    var_diff_by_localhour['KGMainGroup'] = var_diff_by_localhour['KG_ID'].map(kg_main_group_map)
    
    # Calculate average UHI_diff and UWBI_diff by local_hour for each KG main group
    avg_diff_by_hour_and_main_group = var_diff_by_localhour.groupby(['KGMainGroup', 'local_hour'])[['UHI_diff', 'UWBI_diff']].agg(['mean', 'std']).reset_index()
    
    # Find the global minimum and maximum values for UHI_diff and UWBI_diff
    min_uhi_diff = avg_diff_by_hour_and_main_group['UHI_diff'].min()
    max_uhi_diff = avg_diff_by_hour_and_main_group['UHI_diff'].max()
    min_UWBI_diff = avg_diff_by_hour_and_main_group['UWBI_diff'].min()
    max_UWBI_diff = avg_diff_by_hour_and_main_group['UWBI_diff'].max()
    
    # Determine the number of rows needed for subplots
    n_rows = (len(sorted_main_groups) + 3) // 4
    
    # Create subplots
    fig, axs = plt.subplots(n_rows, 4, figsize=(20, 6*n_rows), squeeze=False)
    
    # Plotting
    for i, main_group in enumerate(sorted_main_groups):
        row = i // 4
        col = i % 4
    
        subset = avg_diff_by_hour_and_main_group[avg_diff_by_hour_and_main_group['KGMainGroup'] == main_group]
    
        axs[row, col].plot(subset['local_hour'], subset[('UHI_diff', 'mean')], marker='o', label='UHI_diff Mean')
        
        axs[row, col].fill_between(
            subset['local_hour'],
            subset[('UHI_diff', 'mean')] - subset[('UHI_diff', 'std')],
            subset[('UHI_diff', 'mean')] + subset[('UHI_diff', 'std')],
            color='blue',
            alpha=0.2,
            label='±1 Std Dev'
        )
        
        # Create font dictionaries for the title
        title_font = {'size': 14, 'weight': 'normal'}
        main_group_font = {'size': 16, 'weight': 'bold'}
    
        # Set the title with the main climate group name in bold
        axs[row, col].set_title(f'KG Main Group:', fontdict=title_font)
        axs[row, col].text(0.5, 1.05, main_group, fontdict=main_group_font, transform=axs[row, col].transAxes, ha='center', va='bottom')
    
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
    plt.suptitle('Average Hourly UHI_diff diff by KG Main Group', size=20, weight='bold', y=0.99)  # Add an overall title
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, 'kg_main_group_uhi_diff.png'), dpi=300, bbox_inches='tight')
    plt.close()
