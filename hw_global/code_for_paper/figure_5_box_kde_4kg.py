import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

# ===========================
#    SETUP / DATA LOADING
# ===========================
FilePath = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/'
feather_file = FilePath + 'updated_local_hour_adjusted_variables_HW98.feather'
local_hour_adjusted_df = pd.read_feather(feather_file)

average_by_daytime_and_nighttime = True
if average_by_daytime_and_nighttime:
    # daytime_mask = local_hour_adjusted_df['local_hour'].between(5, 5)
    # # nighttime_mask = (local_hour_adjusted_df['local_hour'].between(20, 24) | local_hour_adjusted_df['local_hour'].between(0, 5))
    # nighttime_mask = local_hour_adjusted_df['local_hour'].between(10, 10)

    daytime_mask = local_hour_adjusted_df["local_hour"].between(8, 16)
    nighttime_mask = local_hour_adjusted_df["local_hour"].between(
        20, 24
    ) | local_hour_adjusted_df["local_hour"].between(0, 4)

    
    daytime_df = local_hour_adjusted_df[daytime_mask]
    nighttime_df = local_hour_adjusted_df[nighttime_mask]
else:
    daytime_df  = local_hour_adjusted_df[ local_hour_adjusted_df["local_hour"] == 16 ].copy()
    nighttime_df= local_hour_adjusted_df[ local_hour_adjusted_df["local_hour"] == 4  ].copy()

# day_loc_mean = (
#     daytime_df.groupby(['location_ID', 'local_hour'])['UHI_diff']
#               .mean()
#               .reset_index()
#               .rename(columns={'UHI_diff': 'mean_UHI_diff'})
# )
# night_loc_mean = (
#     nighttime_df.groupby(['location_ID', 'local_hour'])['UHI_diff']
#                 .mean()
#                 .reset_index()
#                 .rename(columns={'UHI_diff': 'mean_UHI_diff'})
# )

day_loc_mean = (
    daytime_df.groupby(['location_ID'])['UHI_diff']
              .mean()
              .reset_index()
              .rename(columns={'UHI_diff': 'mean_UHI_diff'})
)
night_loc_mean = (
    nighttime_df.groupby(['location_ID'])['UHI_diff']
                .mean()
                .reset_index()
                .rename(columns={'UHI_diff': 'mean_UHI_diff'})
)

# Merge back climate info
day_loc_mean   = pd.merge(
    day_loc_mean,
    daytime_df[['location_ID','KGMajorClass']].drop_duplicates('location_ID'),
    on='location_ID', how='left'
)
night_loc_mean = pd.merge(
    night_loc_mean,
    nighttime_df[['location_ID','KGMajorClass']].drop_duplicates('location_ID'),
    on='location_ID', how='left'
)

# ===========================
#   CLIMATE ZONES + X POS
# ===========================
climate_zones       = ['Arid', 'Tropical', 'Temperate', 'Cold']
climate_zones_disp  = ['Arid', 'Tropical', 'Temperate', 'Continental']  # final labels
pos_dict = {
    'Arid':       0.75,
    'Tropical':   2.35,
    'Temperate':  3.95,
    'Cold':       5.55
}

# ===========================
#    HELPER PLOTTING FUNC
# ===========================
def plot_zone_scatter_box_kde(df, zone, DN, ax, clip0, clip1):
    """
    Jittered scatter, boxplot, and a small inset with a vertical KDE
    in red, anchored at x = pos_dict[zone].
    """
    zone_data = df.loc[df['KGMajorClass']==zone, 'mean_UHI_diff'].dropna()
    y = zone_data.values

    # Print stats
    print(f"[{DN}] {zone} => mean ΔUHI = {y.mean():.2f} °C, N={len(y)}")

    # Jitter scatter
    pos   = pos_dict[zone]
    xs    = np.random.normal(pos, 0.07, size=len(y))
    kde_2d= gaussian_kde(np.vstack([xs,y]))(np.vstack([xs,y]))
    ax.scatter(xs, y, c=np.log(kde_2d), cmap='viridis', s=0.4)

    # Boxplot
    c='black'
    boxprops     = dict(linewidth=1.5, color=c)
    whiskerprops = dict(linewidth=1.5, color=c)
    medianprops  = dict(linewidth=1.5, color='black')
    capprops     = dict(linewidth=1.5, color=c)
    flierprops   = dict(marker=".", markersize=0.8, alpha=0.3, color=c)

    mean_val = np.mean(y)
    std_val  = np.std(y)
    custom_stats = {
        'med': mean_val,
        'q1': mean_val - std_val,
        'q3': mean_val + std_val,
        'whislo': mean_val - 3 * std_val,
        'whishi': mean_val + 3 * std_val,
        'fliers': []
    }
    ax.bxp(
        [custom_stats],
        positions=[pos],
        widths=0.25,
        showfliers=False,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        capprops=capprops,
    )

    # Inset for vertical KDE
    x_offset = 0.35 # the space between the boxplot and the KDE plot
    inset = inset_axes(
        ax,
        width=0.4, 
        height=2.0,
        loc='center',
        bbox_to_anchor=(pos + x_offset, 0.0),  # (x,y) in data coords
        bbox_transform=ax.transData,
        borderpad=0
    )
    inset.set_ylim(clip0, clip1)
    inset.set_xlim(0, 4)
    inset.axis('off')
    sb.kdeplot(y=y, color='red', fill=True, alpha=0.8, linewidth=0,
               clip=(clip0, clip1), ax=inset)
    sb.kdeplot(y=y, color='black', fill=False, linewidth=1,
               clip=(clip0, clip1), common_norm=True, ax=inset)

# ===========================
#      MAIN FIGURE
# ===========================
# Option to display side by side instead of vertically stacked
side_by_side = True  # Set to False for original vertical layout

if side_by_side:
    fig = plt.figure(figsize=(14, 5))
    grid = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[1, 1])
    
    # Create a unified y-range for both plots
    unified_ymin = min(-1.5, -1)  # Take the lowest from both current ranges
    unified_ymax = max(1.5, 2)    # Take the highest from both current ranges
    
    # Use same label height for both plots
    label_height = unified_ymax - 0.3  # Position labels near the top but with some margin
    
    # --------------------------
    #         DAY  (left)
    # --------------------------
    ax_day = fig.add_subplot(grid[0])
    ax_day.spines['top'].set_visible(False)
    ax_day.spines['right'].set_visible(False)
    ax_day.spines['bottom'].set_visible(True)
    
    ax_day.hlines(0, 0, 6, colors='k', linestyles='solid', linewidth=0.8)
    
    # Using unified y-range
    ax_day.set_ylim(unified_ymin, unified_ymax)
    ax_day.set_xlim(0, 6)
    
    ax_day.text(0.1, label_height, 'Day', fontsize=14, weight='bold',
                horizontalalignment='left', verticalalignment='top')
    
    ax_day.set_ylabel('HW-NHW UHI °C', fontsize=14, labelpad=10)
    
    # Plot each zone (Day)
    for z in climate_zones:
        plot_zone_scatter_box_kde(day_loc_mean, z, "Day", ax_day, clip0=-3, clip1=3)
    
    # Set x-axis labels for day plot in side-by-side mode
    x_positions = [pos_dict[z] for z in climate_zones]
    ax_day.set_xticks(x_positions)
    ax_day.set_xticklabels(climate_zones_disp, fontsize=12)
    
    # --------------------------
    #       NIGHT (right)
    # --------------------------
    ax_night = fig.add_subplot(grid[1])
    ax_night.spines['top'].set_visible(False)
    ax_night.spines['right'].set_visible(False)
    ax_night.spines['bottom'].set_visible(True)
    
    ax_night.hlines(0, 0, 6, colors='k', linestyles='solid', linewidth=0.8)
    
    # Using same unified y-range to align scales
    ax_night.set_ylim(unified_ymin, unified_ymax)
    ax_night.set_xlim(0, 6)
    
    ax_night.text(0.1, label_height, 'Night', fontsize=14, weight='bold',
                  horizontalalignment='left', verticalalignment='top')
    
    # Removed y-label from night chart as requested
    
    # Plot each zone (Night)
    for z in climate_zones:
        plot_zone_scatter_box_kde(night_loc_mean, z, "Night", ax_night, clip0=-2, clip1=2)
    
    # Set x-axis labels
    x_positions = [pos_dict[z] for z in climate_zones]
    ax_night.set_xticks(x_positions)
    ax_night.set_xticklabels(climate_zones_disp, fontsize=12)
    
    # Layout
    plt.subplots_adjust(
        top=0.95, bottom=0.15, 
        left=0.08, right=0.95,
        wspace=0.2
    )
    
    outdir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/'
    plt.savefig(outdir + 'Figure_5_box_kde_4KG_side_by_side.png', dpi=600)
    plt.close()

else:
    # Original vertical layout
    fig = plt.figure(figsize=(9,8))
    grid = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1,1])
    
    # --------------------------
    #         DAY  (row 0)
    # --------------------------
    ax_day = fig.add_subplot(grid[0])
    ax_day.spines['top'].set_visible(False)
    ax_day.spines['right'].set_visible(False)
    ax_day.spines['bottom'].set_visible(False)
    
    ax_day.hlines(0, 0, 6, colors='k', linestyles='solid', linewidth=0.8)
    
    # A narrower y‐range for day
    ax_day.set_ylim(-1.5, 1.5)
    ax_day.set_xlim(0, 6)
    
    ax_day.text(0.1, 1.4, 'Day', fontsize=14, weight='bold',
                horizontalalignment='left', verticalalignment='top')
    
    ax_day.set_ylabel('HW-NHW UHI °C', fontsize=14, labelpad=10)
    ax_day.axes.xaxis.set_visible(False)  # hide day x‐axis completely
    
    ax_day.text(-0.6, 1.7, 'a', fontsize=14, weight='bold',
                horizontalalignment='left', verticalalignment='top')
    
    # Plot each zone (Day)
    for z in climate_zones:
        plot_zone_scatter_box_kde(day_loc_mean, z, "Day", ax_day, clip0=-3, clip1=3)
    
    # --------------------------
    #       NIGHT (row 1)
    # --------------------------
    ax_night = fig.add_subplot(grid[1])
    ax_night.spines['top'].set_visible(False)
    ax_night.spines['right'].set_visible(False)
    ax_night.spines['bottom'].set_visible(True)    # we keep a bottom spine for x labels
    
    ax_night.hlines(0, 0, 6, colors='k', linestyles='solid', linewidth=0.8)
    
    # Slightly different y‐range at night
    ax_night.set_ylim(-1, 2)
    ax_night.set_xlim(0, 6)
    
    ax_night.text(0.1, 1.9, 'Night', fontsize=14, weight='bold',
                  horizontalalignment='left', verticalalignment='top')
    ax_night.set_ylabel('HW-NHW UHI °C', fontsize=14, labelpad=10)
    ax_night.text(-0.6, 2.2, 'b', fontsize=14, weight='bold',
                  horizontalalignment='left', verticalalignment='top')
    
    # Plot each zone (Night)
    for z in climate_zones:
        plot_zone_scatter_box_kde(night_loc_mean, z, "Night", ax_night, clip0=-2, clip1=2)
    
    # Now put the climate‐zone labels only on the Night x‐axis
    x_positions = [pos_dict[z] for z in climate_zones]
    ax_night.set_xticks(x_positions)
    ax_night.set_xticklabels(climate_zones_disp, fontsize=12)
    
    # Layout
    plt.subplots_adjust(
        top=0.95, bottom=0.05, 
        left=0.15, right=0.95,
        hspace=0.1
    )
    
    outdir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/'
    plt.savefig(outdir + 'Figure_5_box_kde_4KG.png', dpi=600)
    plt.close()

# Calculate statistics for Day and Night
day_stats = day_loc_mean.groupby('KGMajorClass')['mean_UHI_diff'].agg(['mean', 'std']).reset_index()
night_stats = night_loc_mean.groupby('KGMajorClass')['mean_UHI_diff'].agg(['mean', 'std']).reset_index()

# Merge and rename columns appropriately
stats = pd.merge(day_stats, night_stats, on='KGMajorClass', suffixes=('_Day', '_Night'))
stats = stats.rename(columns={
    'KGMajorClass': 'Climate Zone',
    'mean_Day': 'Day_mean',
    'std_Day': 'Day_std',
    'mean_Night': 'Night_mean',
    'std_Night': 'Night_std'
})

# Map 'Cold' to 'Continental' for consistency with display labels
mapping = {'Arid': 'Arid', 'Tropical': 'Tropical', 'Temperate': 'Temperate', 'Cold': 'Continental'}
stats['Climate Zone'] = stats['Climate Zone'].map(mapping)

# Write the statistics to a CSV file
csv_out = outdir + 'Figure_5_distribution_mean_std.csv'
stats.to_csv(csv_out, index=False)
