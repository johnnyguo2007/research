"""
Plot Figure 5 equivalent for UHI_diff and climate zones, averaged by location_id for day/night hours - CORRECTED KDE and SCALE
"""
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
daytime_df = local_hour_adjusted_df[daytime_mask].copy() # use copy to avoid SettingWithCopyWarning
nighttime_df = local_hour_adjusted_df[nighttime_mask].copy() # use copy to avoid SettingWithCopyWarning

# Group by location_ID and calculate mean UHI_diff for DAY and NIGHT separately
day_location_averaged_df = daytime_df.groupby('location_ID')['UHI_diff'].mean().reset_index()
day_location_averaged_df = day_location_averaged_df.rename(columns={'UHI_diff': 'mean_UHI_diff'})

night_location_averaged_df = nighttime_df.groupby('location_ID')['UHI_diff'].mean().reset_index()
night_location_averaged_df = night_location_averaged_df.rename(columns={'UHI_diff': 'mean_UHI_diff'})

# Merge back KGMajorClass for filtering - DAY
merge_cols_day = ['location_ID', 'KGMajorClass']
day_location_averaged_df = pd.merge(day_location_averaged_df, daytime_df[merge_cols_day].drop_duplicates(subset='location_ID'), on='location_ID', how='left')

# Merge back KGMajorClass for filtering - NIGHT
merge_cols_night = ['location_ID', 'KGMajorClass']
night_location_averaged_df = pd.merge(night_location_averaged_df, nighttime_df[merge_cols_night].drop_duplicates(subset='location_ID'), on='location_ID', how='left')


# Climate Zones to plot
climate_zones = ['Arid', 'Tropical', 'Temperate', 'Cold']
climate_zones_display = ['Arid', 'Tropical', 'Temperate', 'Continental'] # For x axis display

"""
PlotByClimateZone
Description: a function to plot UHI_diff distribution by climate zone, averaged by location_id for specific DN
Input： df -- pandas DataFrame (day_location_averaged_df or night_location_averaged_df)
          Zone -- climate zone name (Arid, Tropical, Temperate, Cold)
          DN -- 'Day' or 'Night' (used for print statement only now)
          ax, inset_ax: axis for boxplot and inset axis for kdeplot
Output: None (plots on the current axes)
"""
def PlotByClimateZone(df, Zone, DN, ax, inset_ax, clip0, clip1): # Modified to accept single inset_ax
    zone_data = df[df['KGMajorClass'] == Zone]['mean_UHI_diff'].dropna() # Use 'mean_UHI_diff'

    if DN=='Day':
        print(f'Climate zone {Zone} Daytime UHI_diff mean (averaged for hour 16): {zone_data.mean()}')
    elif DN=='Night':
        print(f'Climate zone {Zone} Nighttime UHI_diff mean (averaged for hour 4): {zone_data.mean()}')

    c='black'
    flierprops = dict(marker=".",markersize=0.8,alpha=0.3,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=1.5, color=c);whiskerprops=dict( linewidth=1.5, color=c)
    medianprops = dict( linewidth=1.5, color='black');capprops=dict( linewidth=1.5, color=c)

    if Zone == 'Arid':
        pos = 0.75
    elif Zone == 'Tropical':
        pos = 2.35
    elif Zone == 'Temperate':
        pos = 3.95
    elif Zone == 'Cold':
        pos = 5.55
    else:
        raise ValueError("Invalid Climate Zone")

    y = zone_data.values

    # Add some random "jitter" to the x-axis
    x = np.random.normal(pos, 0.07, size=len(y))

    Density1 = np.vstack([x,y])
    Densityz1 = gaussian_kde(Density1)(Density1)
    ax.scatter(x, y,c=np.log(Densityz1),cmap='viridis',s=0.4) #color the scatters by log(density)

    ax.boxplot(y, positions=[pos],whis=(0, 100),widths=0.25,vert=True,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    # plot PDF on an inserted axis:inset_ax
    sb.kdeplot(y=y, color='red',fill=True,alpha=.8, linewidth=0,clip=(clip0,clip1),ax=inset_ax)
    sb.kdeplot(y=y, color='black',fill=False, linewidth=1,clip=(clip0,clip1),common_norm=True,ax=inset_ax)


fig = plt.figure(figsize=(9,8))
heights = [1,1]
spec5 = fig.add_gridspec(ncols=1, nrows=2,height_ratios=heights)

Clip0=-5;Clip1=5 # clip range for day and night plots y axis

DN='Day'
ax = fig.add_subplot(spec5[0])
ax.set_xlim((0,8))
ax.set_ylim((-5, 5)) # Adjusted y limit for UHI_diff
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.hlines(0, 0, 8, colors='k', linestyles='solid',linewidth=0.8)
ax.set_yticks(np.arange(-4, 5, 2))
ax.axes.xaxis.set_visible(False)
# define some properties for the inserted axes
wid=0.6;top=1.025;hgt=2.915#2.87
# this is an inset axes over the main axes
inset1_day = inset_axes(ax, width=wid,height=hgt,
                    bbox_to_anchor=(0.08,top), bbox_transform=ax.transAxes)
inset2_day = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.08 + 1.6,top), bbox_transform=ax.transAxes) # Adjust bbox_to_anchor for each inset
inset3_day = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.08 + 1.6*2,top), bbox_transform=ax.transAxes) # Adjust bbox_to_anchor for each inset
inset4_day = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.08 + 1.6*3,top), bbox_transform=ax.transAxes) # Adjust bbox_to_anchor for each inset


inset_axes_day = [inset1_day, inset2_day, inset3_day, inset4_day]
for inset_ax in inset_axes_day:
    inset_ax.set_ylim((-5, 5)); # Adjusted y limit for UHI_diff
    inset_ax.set_xlim((0,0.8));
    inset_ax.axis('off');


ax.text(0.1,5.3,'Day',horizontalalignment='left',verticalalignment='top',fontsize=14,weight='bold')
ax.set_ylabel('ΔUHI (°C)',fontsize=14,labelpad=-9) # Modified y axis label

for i, zone in enumerate(climate_zones):
    PlotByClimateZone(day_location_averaged_df, zone, 'Day', ax, inset_axes_day[i], Clip0, Clip1) # using same inset for both plot in each column

ax.tick_params(axis='both',labelsize=14,direction='in')
ax.text(-0.3,5.5,'a',horizontalalignment='left',verticalalignment='top',fontsize=14,weight='bold')


DN='Night'
Clip0=-3;Clip1=3 # wider clip range for night plots y axis
ax = fig.add_subplot(spec5[1])
ax.set_xlim((0,8))
ax.set_ylim((-3, 3)) # Adjusted y limit for UHI_diff Night - wider scale
ax.hlines(0, 0, 8, colors='k', linestyles='solid',linewidth=0.8)
ax.axes.xaxis.set_visible(False)
ax.set_yticks(np.arange(-2, 4, 1))

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
#define some properties for the inserted axes

wid=0.6;hgt=2.915;xlim0=0;xlim1=5;ylim0=-3;ylim1=3 # Adjusted y limit for UHI_diff Night - wider scale
# this is an inset axes over the main axes
inset1_night = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.08,top), bbox_transform=ax.transAxes)
inset2_night = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.08 + 1.6,top), bbox_transform=ax.transAxes) # Adjust bbox_to_anchor for each inset
inset3_night = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.08 + 1.6*2,top), bbox_transform=ax.transAxes) # Adjust bbox_to_anchor for each inset
inset4_night = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.08 + 1.6*3,top), bbox_transform=ax.transAxes) # Adjust bbox_to_anchor for each inset


inset_axes_night = [inset1_night, inset2_night, inset3_night, inset4_night]
for inset_ax in inset_axes_night:
    inset_ax.set_ylim((ylim0,ylim1)); # Adjusted y limit for UHI_diff Night - wider scale
    inset_ax.set_xlim((xlim0,xlim1));
    inset_ax.axis('off');


ax.text(0.1,2.9,'Night',horizontalalignment='left',verticalalignment='top',fontsize=14,weight='bold')
ax.set_ylabel('ΔUHI (°C)',fontsize=14,labelpad=-9) # Modified y axis label

for i, zone in enumerate(climate_zones):
    PlotByClimateZone(night_location_averaged_df, zone, 'Night', ax, inset_axes_night[i], Clip0, Clip1) # using same inset for each plot in each column


ax.tick_params(axis='both',labelsize=14,direction='in')
ax.text(-0.3,3.1,'b',horizontalalignment='left',verticalalignment='top',fontsize=14,weight='bold')

# create a white rectangle to cover y-axis if needed, not needed for now as y axis range is smaller

# add x-axis text annotations
baseline=-2.8 # Adjusted baseline for night plot
b=np.arange(ax.get_xlim()[0],ax.get_xlim()[1]*0.93,0.1)
ax.plot(b, b*0+baseline,'k',linewidth=1.5)
x_positions = [0.75, 2.35, 3.95, 5.55]
ax.scatter(x_positions, [baseline+0.35]*len(x_positions),marker="^",s=14,c='k')

for i, zone_display in enumerate(climate_zones_display):
    ax.text(x_positions[i], baseline+2.8, zone_display, horizontalalignment='center', verticalalignment='top', fontsize=14)

# Climate zone labels below x axis
baseline_climate_label = baseline - 0.36
ax.text(x_positions[0], baseline_climate_label, climate_zones_display[0], horizontalalignment='center', fontweight="bold", verticalalignment='top', fontsize=14)
ax.text(x_positions[1], baseline_climate_label, climate_zones_display[1], horizontalalignment='center', fontweight="bold", verticalalignment='top', fontsize=14)
ax.text(x_positions[2], baseline_climate_label, climate_zones_display[2], horizontalalignment='center', fontweight="bold", verticalalignment='top', fontsize=14)
ax.text(x_positions[3], baseline_climate_label, 'Cont.', horizontalalignment='center', fontweight="bold", verticalalignment='top', fontsize=14) # Abbreviated Continental

# Adjust x tick positions for climate zones
ax.set_xticks([0.75, 2.35, 3.95, 5.55])
ax.set_xticklabels(climate_zones_display, fontsize=12) # Set x tick labels
ax.tick_params(axis='x', which='major', pad=7) # Adjust padding for x tick labels
ax.spines['bottom'].set_visible(True) # Make bottom spine visible for x axis


plt.subplots_adjust(top=0.9,
bottom=0.15, # Adjusted bottom to make space for x axis labels
left=0.11,
right=0.9,
hspace=0.23,
wspace=0.21)

output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/'
plt.savefig(output_dir+'Figure_3_distribution_4KG.png', dpi=600)
plt.close()