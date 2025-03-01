# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

import sys
sys.path.append('/home/jguo/research/hw_global/ultimate/')
# Import get_latex_label from plot_util
from mlflow_tools.plot_util import get_latex_label
# Load your data
# Path to your Feather file
explain_variance_by_kg_duration_path = "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/paper_data/explain_variance_by_kg_duration.feather"

if not os.path.exists(explain_variance_by_kg_duration_path):

    feather_path = "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/updated_local_hour_adjusted_variables_HW98.feather"

    # Read the Feather file into a DataFrame
    df = pd.read_feather(feather_path)

    # 1. Calculate Daily Average
    df['date'] = df['time'].dt.date  # Extract date


    # 3. Sort so day_in_event is computed in the correct chronological order
    df = df.sort_values(['location_ID', 'event_ID', 'time'])

    # 4. Compute "day_in_event"
    df['day_in_event'] = df.groupby(['location_ID', 'event_ID'])['date']\
                        .transform(lambda x: (pd.to_datetime(x) - pd.to_datetime(x.min())).dt.days)

    # 5. Aggregate data to daily level (mean of UHI_diff, Q2M, SOILWATER_10CM)
    df_agg = (df.groupby(['KGMajorClass', 'day_in_event'], as_index=False)
                .agg(UHI_diff_mean=('UHI_diff','mean'),
                    UHI_diff_std=('UHI_diff','std'),
                    Q2M_mean=('Q2M', 'mean'),
                    SOILWATER_10CM_mean=('SOILWATER_10CM', 'mean')))
  
    df_agg.to_feather(explain_variance_by_kg_duration_path)  
else:
    df_agg = pd.read_feather(explain_variance_by_kg_duration_path)


# --- REMOVE POLAR ZONE ---
df_plot = df_agg[df_agg['KGMajorClass'] != 'Polar'].copy()

# --- FILTER FOR FIRST N DAYS ---
N_DAYS = 11
df_plot = df_plot[df_plot['day_in_event'] < N_DAYS].copy()

# --- LAYOUT OPTIONS ---
# Add command line arguments if running as script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot heatwave data with different layout options')
    parser.add_argument('--grid', action='store_true', help='Use 2x2 grid layout instead of one row')
    args = parser.parse_args()
    
    # Command line args take precedence if provided
    if 'args' in locals():
        use_grid_layout = args.grid
    else:
        use_grid_layout = False  # Default to one row when no args
else:
    # Default to one row when running in notebook
    use_grid_layout = False

# --- LAYOUT OPTION ---
one_row_layout = False  # Set to True to put all plots in one row, False for 2x2 grid

# Override with our preferred default (one row)
one_row_layout = not use_grid_layout  # True for one row, False for 2x2 grid

# 6. Set up the plot layout based on the configuration
unique_zones = sorted(df_plot['KGMajorClass'].dropna().unique())  # sort & drop NA
n_zones = len(unique_zones)

# Set rows and columns based on layout preference
if one_row_layout:
    ncols = n_zones
    nrows = 1
else:
    ncols = 2
    nrows = 2

# ### Normalize Q2M and SOILWATER_10CM to a 0-1 scale (Min-Max scaling) - GLOBAL Normalization

# Find GLOBAL min and max for Q2M and SOILWATER_10CM from df_plot (before further grouping)
min_max_values = {}
for var in ['Q2M_mean', 'SOILWATER_10CM_mean']: # Only normalize Q2M and SOILWATER_10CM
    min_val = df_plot[var].min()  # Use df_plot to calculate GLOBAL min
    max_val = df_plot[var].max()  # Use df_plot to calculate GLOBAL max
    min_max_values[var] = {'min': min_val, 'max': max_val}

# Further aggregate for plotting - AFTER global min/max calculation
df_plot_agg = (
    df_plot.groupby(['KGMajorClass', 'day_in_event'], as_index=False)
                    .agg(
                   UHI_diff_mean=('UHI_diff_mean','mean'),
                   UHI_diff_std=('UHI_diff_std','mean'),  # simple approach: averaging standard deviations
                   Q2M_mean=('Q2M_mean', 'mean'),
                   SOILWATER_10CM_mean=('SOILWATER_10CM_mean', 'mean')
               )
)


# Apply Min-Max scaling to Q2M and SOILWATER_10CM using GLOBAL min/max
for var in ['Q2M_mean', 'SOILWATER_10CM_mean']: # Only normalize Q2M and SOILWATER_10CM
    df_plot_agg[f'{var}_scaled'] = df_plot_agg.apply(
        lambda row: (row[var] - min_max_values[var]['min']) / (min_max_values[var]['max'] - min_max_values[var]['min']),
        axis=1
    )

# Find global min/max for UHI_diff for consistent y-axis scaling
uhi_min = df_plot_agg['UHI_diff_mean'].min()
uhi_max = df_plot_agg['UHI_diff_mean'].max()

# Set up subplot grid
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(5*ncols, 4*nrows),  # Adjust figure size based on layout
                         sharex=True, sharey=True)
axes = axes.flatten() if n_zones > 1 else [axes]

colors = ['C0', 'C1', 'C2']
variables_left_yaxis = ['UHI_diff']
variables_right_yaxis = ['Q2M', 'SOILWATER_10CM']
variable_means_left = ['UHI_diff_mean']
variable_means_right_scaled = ['Q2M_mean_scaled', 'SOILWATER_10CM_mean_scaled']

for i, zone in enumerate(unique_zones):
    if i < nrows * ncols:
        ax = axes[i]

        # Subset data for this zone
        sub_agg = df_plot_agg[df_plot_agg['KGMajorClass'] == zone].copy()

        # Plot UHI_diff on the primary (left) y-axis
        for var_index, var in enumerate(variables_left_yaxis):
            mean_var_col = variable_means_left[var_index]
            line = ax.plot(
                sub_agg['day_in_event'],
                sub_agg[mean_var_col],
                label=f"{get_latex_label(var)} (°C)",
                color=colors[0],
                marker='o',
                markersize=3,
                linestyle='-'
            )[0]
                
        # Only add left y-axis label to the first plot in one-row layout, or to all plots in grid layout
        if not one_row_layout or (one_row_layout and i == 0):
            ax.set_ylabel("HW-NHW UHI (°C)", color=colors[0])
        
        ax.tick_params(axis='y', labelcolor=colors[0])
        # Set consistent y-axis limits for UHI_diff
        ax.set_ylim(uhi_min, uhi_max)

        # Create secondary y-axis, sharing x-axis
        ax_right = ax.twinx()

        # Plot normalized Q2M and SOILWATER_10CM on the secondary (right) y-axis
        for var_index, var in enumerate(variables_right_yaxis):
            mean_var_col_scaled = variable_means_right_scaled[var_index]
            line = ax_right.plot(
                sub_agg['day_in_event'],
                sub_agg[mean_var_col_scaled],
                label=f"{get_latex_label(var)} (normalized)",
                color=colors[var_index+1],
                marker='o',
                markersize=3,
                linestyle='-'
            )[0]
                
        # Only add right y-axis label to the last plot in one-row layout, or to all plots in grid layout
        if not one_row_layout or (one_row_layout and i == n_zones-1):
            ax_right.set_ylabel("Globally Normalized " + 
                            f"{get_latex_label('Q2M')} & {get_latex_label('SOILWATER_10CM')} (0-1)", 
                            color='black')
        
        ax_right.tick_params(axis='y', labelcolor='black')
        # Set consistent y-axis limits for normalized variables
        ax_right.set_ylim(0, 1)  # Since these are normalized values, they should be between 0 and 1

        # Add legend based on layout
        if (one_row_layout and i == 0) or (not one_row_layout and i == 0):  # First plot
            lines_left, labels_left = ax.get_legend_handles_labels()
            lines_right, labels_right = ax_right.get_legend_handles_labels()
            ax.legend(lines_left + lines_right, labels_left + labels_right, 
                     loc='upper right')
        elif (one_row_layout and i == n_zones-1) or (not one_row_layout and i == 3):  # Last plot
            lines_left, labels_left = ax.get_legend_handles_labels()
            lines_right, labels_right = ax_right.get_legend_handles_labels()
            ax.legend(lines_left + lines_right, labels_left + labels_right, 
                     loc='lower left')

        ax.set_title(zone)
        ax.set_xlabel("Day in Heatwave Event")
        ax.set_xlim(0, N_DAYS - 1)

    else:
        break

# Turn off extra axes
for j in range(i + 1, nrows*ncols):
    if j < len(axes):
        fig.delaxes(axes[j])

# plt.suptitle(f"Day-by-Day Changes of {get_latex_label('UHI_diff')}, Globally Normalized {get_latex_label('Q2M')} & {get_latex_label('SOILWATER_10CM')} by Climate Zone (First 10 Days)", 
#              fontsize=14, y=0.95)
plt.tight_layout()
plt.show()