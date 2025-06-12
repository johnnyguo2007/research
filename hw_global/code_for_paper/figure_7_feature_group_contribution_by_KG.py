import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/jguo/research/hw_global/ultimate/')
from mlflow_tools.plot_util import replace_cold_with_continental

# Add lookup table reading
lookup_df = pd.read_excel('/home/jguo/research/hw_global/Data/var_name_unit_lookup.xlsx')
lookup_dict = dict(zip(lookup_df['Variable'], lookup_df['LaTeX']))

# Define the output directory
# output_dir = '/home/jguo/tmp/output'
output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper'
os.makedirs(output_dir, exist_ok=True)

# Update base directory to point to the new data_only_24_hourly location
base_directory = "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts/893793682234305734/67f2e168085e4507b0a79941b74d7eb7/artifacts/data_only_24_hourly"

# Define climate zones - note that files use Continental instead of Cold
climate_zones = ["Arid", "Cold", "Temperate", "Tropical"]
file_climate_zones = ["Arid", "Continental", "Temperate", "Tropical"]  # For file paths

# Define Köppen-Geiger zone colors based on the provided table
major_zone_colors = {
    "Tropical": "#0000FF",    # blue tones for Tropical climates
    "Arid": "#FF0000",        # red tones for Arid climates
    "Temperate": "#008000",   # green tones for Temperate climates
    "Cold": "#DDA0DD",        # Light purple tones for Continental (Cold) climates
    "Polar": "#D3D3D3",       # Light gray tones for Polar climates
}

# Function to create and save a plot for a specific time period
def create_feature_group_plot(time_period, ax):
    """
    Create and save a feature group contribution plot for the specified time period.
    
    Args:
        time_period: Either 'day' or 'night'
        ax: Matplotlib Axes object to draw the plot on.
    """
    # Initialize an empty DataFrame to consolidate data
    consolidated_data = pd.DataFrame()
    
    # Map time_period to directory name
    time_dir = f"{time_period}time_group_summary_plots"
    
    # Read each CSV file and append it to the consolidated DataFrame
    for zone, file_zone in zip(climate_zones, file_climate_zones):
        # Construct the new file name
        new_file_name = f"group_importance_{file_zone}_{time_period}_group_shap_importance.csv"
        file_path = os.path.join(
            base_directory, 
            time_dir, 
            file_zone, 
            new_file_name
        )
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path) # Read CSV without index_col
            
            # The new CSV has 'Group' and 'Percentage' columns.
            # 'Percentage' is already calculated correctly.
            if not df.empty and 'Group' in df.columns and 'Percentage' in df.columns:
                # Create a DataFrame with the results
                zone_df = pd.DataFrame({
                    'Feature Group': df['Group'],
                    'Percentage': df['Percentage'],
                    'Region': zone  # Use original zone name for consistency
                })
                consolidated_data = pd.concat([consolidated_data, zone_df], ignore_index=True)
            else:
                # Update warning message for new format
                print(f"Warning: Required columns ('Group', 'Percentage') not found or file is empty in {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    if consolidated_data.empty:
        print(f"No data found for {time_period} period")
        return
    
    # Filter out FSA features for nighttime plot
    if time_period == 'night':
        consolidated_data = consolidated_data[~consolidated_data['Feature Group'].str.contains('FSA', case=False, na=False)]

    # Pivot the data for grouped bar chart
    data_pivot = consolidated_data.pivot(index='Feature Group', columns='Region', values='Percentage')
    
    # Use consistent colors for the bar chart. This must be done before renaming the columns.
    colors = [major_zone_colors[zone] for zone in data_pivot.columns]
    
    # Rename the columns to replace 'Cold' with 'Continental' for display purposes.
    data_pivot.columns = [replace_cold_with_continental(zone) for zone in data_pivot.columns]
    
    # Map feature groups to LaTeX labels
    latex_index = []
    for feature_group in data_pivot.index:
        latex_label = lookup_dict.get(feature_group)  # Get the LaTeX label
        # Use original value if latex_label is None, empty string, or NaN
        if pd.isna(latex_label) or latex_label == '':
            latex_label = feature_group
        latex_index.append(latex_label)
    data_pivot.index = latex_index

    # Sort feature groups alphabetically
    data_pivot = data_pivot.sort_index()
    
    # Plot the grouped bar chart
    data_pivot.plot(kind='bar', ax=ax, width=0.75, color=colors)
    
    # Add gridlines for better readability
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set plot title and labels
    time_label = "Daytime" if time_period == "day" else "Nighttime"
    ax.set_title(f'{time_label}', fontsize=20, weight='bold', pad=20)
    ax.set_xlabel('Feature Group', fontsize=16, labelpad=10)
    ax.set_ylabel('Percentage Contribution (%)', fontsize=16, labelpad=10)
    
    # Customize x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Adjust legend to align with the defined regions and colors
    ax.legend(
        title='Region',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=12,
        title_fontsize=14,
    )
    
    # Add labels only to the bars with the highest value for each feature group
    for i, feature_group in enumerate(data_pivot.index):
        # Get the maximum value and its corresponding region
        max_value = data_pivot.loc[feature_group].max()
        max_region = data_pivot.columns[data_pivot.loc[feature_group] == max_value][0]
        
        # Find the index of the max_region and add label to its corresponding bar
        region_index = data_pivot.columns.get_loc(max_region)
        bar_container = ax.containers[region_index]
        
        # Label only the bar with the highest percentage
        ax.bar_label(
            bar_container,
            labels=[f'{max_value:.1f}%' if j == i else '' for j in range(len(data_pivot))],
            label_type='edge',
            fontsize=12,
            fontweight='bold',
        )
    
    # Improve layout to avoid overlapping elements
    plt.tight_layout()
    
    # Save the plot
    # output_filename = f'Figure_7_feature_group_contribution_by_KG_{time_period}.png'
    # plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight', dpi=600)
    # plt.close()
    
    # print(f"Saved {output_filename}")

# Function to generate and save a single plot (day or night)
def generate_single_plot(time_period):
    """
    Generates and saves a single feature group contribution plot for the specified time period.
    Args:
        time_period: Either 'day' or 'night'
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    create_feature_group_plot(time_period, ax) # Call the main plotting function

    # Improve layout
    plt.tight_layout()

    # Save the plot
    output_filename = f'Figure_7_feature_group_contribution_by_KG_{time_period}.png'
    fig.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight', dpi=600)
    plt.close(fig)
    
    print(f"Saved {output_filename}")


# Generate plots for both day and night
# create_feature_group_plot('day') # Old direct calls
# create_feature_group_plot('night') # Old direct calls

def create_combined_feature_group_plot():
    """
    Creates a combined plot with side-by-side feature group contributions for day and night.
    """
    fig, axes = plt.subplots(1, 2, figsize=(34, 12)) # Adjusted figsize for two plots

    # Create day plot
    create_feature_group_plot('day', axes[0])
    
    # Create night plot
    create_feature_group_plot('night', axes[1])

    # Remove individual legends if they are identical and add a shared one
    handles, labels = axes[0].get_legend_handles_labels()
    # Use display_regions for labels in the shared legend
    # We need to ensure display_regions is consistent or derived correctly for the shared legend
    # For now, let's assume the labels from axes[0] are representative
    # And that climate_zones (which determines colors) and display_regions are what we want
    
    # Get the display names for the legend from the first plot's legend
    # This assumes the regions are the same for both plots, which should be the case.
    # The `labels` variable from `axes[0].get_legend_handles_labels()` already gives us this.

    for ax in axes:
        if ax.get_legend() is not None: # Check if legend exists before trying to remove
            ax.get_legend().remove()
            
    # Use the `labels` obtained from `axes[0].get_legend_handles_labels()`
    # The `handles` are also from `axes[0]`, which should be fine.
    fig.legend(handles, labels, 
               loc='upper center', # Position the legend at the top center
               bbox_to_anchor=(0.5, 0.97), # Adjust anchor to place it neatly below suptitle
               ncol=len(labels), # Number of columns based on number of regions
               title='Region', 
               fontsize=12, 
               title_fontsize=14,
               frameon=False # Optional: remove frame for a cleaner look
    )

    # Set overall title
    fig.suptitle('Percentage Contribution of Feature Groups by Region', fontsize=24, weight='bold', y=1.03)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to make space for suptitle and shared legend

    # Save the combined plot
    output_filename = 'Figure_7_feature_group_contribution_by_KG_combined.png'
    fig.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight', dpi=600)
    plt.close(fig)
    
    print(f"Saved {output_filename}")

# Generate individual plots first
generate_single_plot('day')
generate_single_plot('night')

# Generate the combined plot
create_combined_feature_group_plot()
