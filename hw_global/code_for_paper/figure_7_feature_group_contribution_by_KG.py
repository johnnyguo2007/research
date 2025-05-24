import pandas as pd
import matplotlib.pyplot as plt
import os

def replace_cold_with_continental(kg_main_group):
    if kg_main_group == 'Cold':
        return 'Continental'
    return kg_main_group

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
def create_feature_group_plot(time_period):
    """
    Create and save a feature group contribution plot for the specified time period.
    
    Args:
        time_period: Either 'day' or 'night'
    """
    # Initialize an empty DataFrame to consolidate data
    consolidated_data = pd.DataFrame()
    
    # Map time_period to directory name
    time_dir = f"{time_period}time_group_summary_plots"
    
    # Read each CSV file and append it to the consolidated DataFrame
    for zone, file_zone in zip(climate_zones, file_climate_zones):
        file_path = os.path.join(
            base_directory, 
            time_dir, 
            file_zone, 
            f"group_importance_{file_zone}_{time_period}_group_shap_data.csv"
        )
        
        if os.path.exists(file_path):
            # Read the CSV - the data is transposed, so we need to handle it properly
            df = pd.read_csv(file_path, index_col=0)
            
            # The data has groups as columns and 'mean_shap' as the row
            # We need to transpose it and calculate percentages
            if 'mean_shap' in df.index:
                shap_values = df.loc['mean_shap']
                
                # Remove metadata columns
                metadata_cols = ['base_value', 'UHI_diff', 'y_pred', 'Estimation_Error']
                shap_values = shap_values.drop(metadata_cols, errors='ignore')
                
                # Calculate percentages
                total_shap = shap_values.abs().sum()
                if total_shap > 0:
                    percentages = (shap_values.abs() / total_shap) * 100
                    
                    # Create a DataFrame with the results
                    zone_df = pd.DataFrame({
                        'Feature Group': percentages.index,
                        'Percentage': percentages.values,
                        'Region': zone  # Use original zone name for consistency
                    })
                    
                    consolidated_data = pd.concat([consolidated_data, zone_df], ignore_index=True)
            else:
                print(f"Warning: 'mean_shap' row not found in {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    if consolidated_data.empty:
        print(f"No data found for {time_period} period")
        return
    
    # Pivot the data for grouped bar chart
    data_pivot = consolidated_data.pivot(index='Feature Group', columns='Region', values='Percentage')
    
    # Apply replacement to Regions for plotting
    display_regions = [replace_cold_with_continental(zone) for zone in data_pivot.columns]
    
    # Use consistent colors for the bar chart
    colors = [major_zone_colors[zone] for zone in data_pivot.columns]
    
    # Map feature groups to LaTeX labels
    latex_index = []
    for feature_group in data_pivot.index:
        latex_label = lookup_dict.get(feature_group)  # Get the LaTeX label
        # Use original value if latex_label is None, empty string, or NaN
        if pd.isna(latex_label) or latex_label == '':
            latex_label = feature_group
        latex_index.append(latex_label)
    data_pivot.index = latex_index
    
    # Plot the grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 10))
    data_pivot.plot(kind='bar', ax=ax, width=0.75, color=colors)
    
    # Add gridlines for better readability
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set plot title and labels
    time_label = "Daytime" if time_period == "day" else "Nighttime"
    ax.set_title(f'Percentage Contribution of Feature Groups by Region ({time_label})', 
                 fontsize=20, weight='bold', pad=20)
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
        labels=display_regions,
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
    output_filename = f'Figure_7_feature_group_contribution_by_KG_{time_period}.png'
    plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight', dpi=600)
    plt.close()
    
    print(f"Saved {output_filename}")

# Generate plots for both day and night
create_feature_group_plot('day')
create_feature_group_plot('night')
