import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")

# FilePath for the feather file
FilePath = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/'
feather_file = FilePath + 'updated_local_hour_adjusted_variables_HW98.feather'
local_hour_adjusted_df = pd.read_feather(feather_file)

# Define daytime and nighttime masks
daytime_mask = local_hour_adjusted_df["local_hour"] == 16
nighttime_mask = local_hour_adjusted_df["local_hour"] == 4

# Separate dataframes for day and night
daytime_df = local_hour_adjusted_df[daytime_mask].copy()
nighttime_df = local_hour_adjusted_df[nighttime_mask].copy()

# Group by location_ID and calculate mean UHI_diff for DAY and NIGHT
day_location_averaged_df = (
    daytime_df.groupby('location_ID')['UHI_diff'].mean().reset_index()
               .rename(columns={'UHI_diff': 'mean_UHI_diff'})
)
night_location_averaged_df = (
    nighttime_df.groupby('location_ID')['UHI_diff'].mean().reset_index()
                .rename(columns={'UHI_diff': 'mean_UHI_diff'})
)

# Merge back KGMajorClass for day
day_location_averaged_df = pd.merge(
    day_location_averaged_df,
    daytime_df[['location_ID','KGMajorClass']].drop_duplicates('location_ID'),
    on='location_ID', how='left'
)

# Merge back KGMajorClass for night
night_location_averaged_df = pd.merge(
    night_location_averaged_df,
    nighttime_df[['location_ID','KGMajorClass']].drop_duplicates('location_ID'),
    on='location_ID', how='left'
)

# Climate zones
climate_zones       = ['Arid', 'Tropical', 'Temperate', 'Cold']
climate_zones_disp  = ['Arid', 'Tropical', 'Temperate', 'Continental']
pos_dict = {
    'Arid':       0.75,
    'Tropical':   2.35,
    'Temperate':  3.95,
    'Cold':       5.55
}

def PlotByClimateZone(df, zone, DN, ax, clip0, clip1):
    """
    Plots scatter+box in `ax` at x=pos_dict[zone], plus an inset axes
    for the red KDE to the right of the box.
    """
    zone_data = df[df['KGMajorClass'] == zone]['mean_UHI_diff'].dropna()
    y = zone_data.values
    
    # Print stats
    if DN == 'Day':
        print(f"[DAY] {zone} => mean ΔUHI: {zone_data.mean():.2f} °C")
    else:
        print(f"[NIGHT] {zone} => mean ΔUHI: {zone_data.mean():.2f} °C")

    # x-position for the box
    pos = pos_dict[zone]

    # Scatter points with density coloring
    x_scatter = np.random.normal(pos, 0.07, size=len(y))  # jitter
    density   = gaussian_kde(np.vstack([x_scatter,y]))(np.vstack([x_scatter,y]))
    ax.scatter(
        x_scatter, y, c=np.log(density), cmap='viridis', s=0.4, alpha=1
    )

    # Boxplot
    c='black'
    flierprops   = dict(marker=".", markersize=0.8, alpha=0.3, color=c, markeredgecolor=c)
    boxprops     = dict(linewidth=1.5, color=c)
    whiskerprops = dict(linewidth=1.5, color=c)
    medianprops  = dict(linewidth=1.5, color='black')
    capprops     = dict(linewidth=1.5, color=c)

    ax.boxplot(
        y, positions=[pos], whis=(0, 100), widths=0.25, vert=True,
        flierprops=flierprops, capprops=capprops, medianprops=medianprops,
        boxprops=boxprops, whiskerprops=whiskerprops
    )

    # Make an inset axis for the KDE *to the right* of the box in data coords
    x_offset = 0.3  # how far from the box to put the KDE
    inset = inset_axes(
        ax,
        width=0.4,   # tweak to your liking
        height=2.0,  # tweak to your liking
        loc='center',  # position "inside" the bounding box
        bbox_to_anchor=(pos + x_offset, 0.0),  # x,y in data coords
        bbox_transform=ax.transData,           # interpret in data coords
        borderpad=0
    )
    inset.set_ylim((clip0, clip1))
    inset.set_xlim((0, 1))    # let the KDE fill horizontally
    inset.axis('off')

    # Plot the KDE in the inset
    sb.kdeplot(y=y, color='red', fill=True, alpha=0.8, linewidth=0,
               clip=(clip0, clip1), ax=inset)
    sb.kdeplot(y=y, color='black', fill=False, linewidth=1,
               clip=(clip0, clip1), common_norm=True, ax=inset)


fig = plt.figure(figsize=(9,8))
heights = [1,1]
spec5 = fig.add_gridspec(ncols=1, nrows=2, height_ratios=heights)

# --- DAY subplot ---
ax_day = fig.add_subplot(spec5[0])
ax_day.set_xlim((0,8))
ax_day.set_ylim((-5, 5))
ax_day.spines['top'].set_visible(False)
ax_day.spines['bottom'].set_visible(False)
ax_day.spines['right'].set_visible(False)
ax_day.hlines(0, 0, 8, colors='k', linestyles='solid', linewidth=0.8)
ax_day.set_yticks(np.arange(-4, 5, 2))
ax_day.axes.xaxis.set_visible(False)
ax_day.text(0.1, 5.3, 'Day', horizontalalignment='left', verticalalignment='top',
            fontsize=14, weight='bold')
ax_day.set_ylabel('ΔUHI (°C)', fontsize=14, labelpad=-9)
ax_day.tick_params(axis='both', labelsize=14, direction='in')
ax_day.text(-0.3, 5.5, 'a', horizontalalignment='left', verticalalignment='top',
            fontsize=14, weight='bold')

# Plot each climate zone’s data (daytime)
clip0_day, clip1_day = -5, 5
for z in climate_zones:
    PlotByClimateZone(day_location_averaged_df, z, 'Day', ax_day, clip0_day, clip1_day)

# --- NIGHT subplot ---
ax_night = fig.add_subplot(spec5[1])
ax_night.set_xlim((0,8))
ax_night.set_ylim((-3, 3))
ax_night.hlines(0, 0, 8, colors='k', linestyles='solid', linewidth=0.8)
ax_night.set_yticks(np.arange(-2, 4, 1))
ax_night.spines['top'].set_visible(False)
ax_night.spines['bottom'].set_visible(False)
ax_night.spines['right'].set_visible(False)
ax_night.text(0.1, 2.9, 'Night', horizontalalignment='left', verticalalignment='top',
              fontsize=14, weight='bold')
ax_night.set_ylabel('ΔUHI (°C)', fontsize=14, labelpad=-9)
ax_night.tick_params(axis='both', labelsize=14, direction='in')
ax_night.text(-0.3, 3.1, 'b', horizontalalignment='left', verticalalignment='top',
              fontsize=14, weight='bold')

clip0_night, clip1_night = -3, 3
for z in climate_zones:
    PlotByClimateZone(night_location_averaged_df, z, 'Night', ax_night, clip0_night, clip1_night)

# Add climate‐zone labels on the bottom (arrows, text, etc.)
baseline = -2.8
b = np.arange(ax_night.get_xlim()[0], ax_night.get_xlim()[1]*0.93, 0.1)
ax_night.plot(b, b*0 + baseline,'k', linewidth=1.5)

x_positions = [pos_dict[z] for z in climate_zones]
ax_night.scatter(x_positions, [baseline+0.35]*len(x_positions),
                 marker="^", s=14, c='k')
for i, zone_label in enumerate(climate_zones_disp):
    ax_night.text(x_positions[i], baseline+2.8, zone_label,
                  horizontalalignment='center', verticalalignment='top',
                  fontsize=14)

# Also re‐label beneath the axis
baseline_climate_label = baseline - 0.36
for i, zone_label in enumerate(climate_zones_disp):
    ax_night.text(x_positions[i], baseline_climate_label, zone_label,
                  horizontalalignment='center', fontweight="bold",
                  verticalalignment='top', fontsize=14)

ax_night.set_xticks(x_positions)
ax_night.set_xticklabels(climate_zones_disp, fontsize=12)
ax_night.tick_params(axis='x', which='major', pad=7)
ax_night.spines['bottom'].set_visible(True)

plt.subplots_adjust(top=0.9, bottom=0.15, left=0.11, right=0.9,
                    hspace=0.23, wspace=0.21)

output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/'
plt.savefig(output_dir + 'Figure_3_distribution_4KG.png', dpi=600)
plt.close()
