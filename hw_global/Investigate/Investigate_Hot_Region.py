# %%
import pandas as pd
import numpy as np
import xarray as xr
import os

# %%
summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'
# Load data
merged_feather_path = os.path.join(summary_dir, 'local_hour_adjusted_variables_with_location_ID_event_ID_and_sur.feather')
local_hour_adjusted_df = pd.read_feather(merged_feather_path)

# %%
local_hour_adjusted_df['URBAN_HEAT'].hist()

# %%
#group by global event ID and count the number of days in each event
event_count = local_hour_adjusted_df.groupby('location_ID')['time'].count().sort_values(ascending=False)
event_count

# %%

location_hw_count = (
    local_hour_adjusted_df[['lat', 'lon']]
    .groupby(['lat', 'lon'])
    .size()
    .reset_index(name='count')
    # .query('count / (24 *29) > 40')
)

# Divide count by 24 to get the number of days
location_hw_count['count'] = location_hw_count['count'] / (24 *29)


location_hw_count.reset_index(drop=True, inplace=True)


# %%
location_hw_count[location_hw_count['count'] > 60].info()

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def normalize_longitude(lons):
    """
    Normalize longitudes to be within the range [-180, 180].
    """
    normalized_lons = np.where(lons > 180, lons - 360, lons)
    return normalized_lons


def draw_map_subplot(ax, title, data, variable):
    if data.empty:
        print(f"No data available for {title}. Skipping plot.")
        ax.set_title(title + " - No Data Available")
        return

    m = Basemap(projection='cyl', lon_0=0, ax=ax,
                fix_aspect=False,
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180)
    m.drawcoastlines(color='0.15', linewidth=0.5, zorder=3)
    m.drawcountries(linewidth=0.1)
    m.fillcontinents(color='white', lake_color='lightcyan')
    m.drawmapboundary(fill_color='lightcyan')
    m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1], fontsize=10)

    normalized_lons = normalize_longitude(data['lon'].values)
    x, y = m(normalized_lons, data['lat'].values)

    vmin, vmax = data[variable].min(), data[variable].max()
    sc = m.scatter(x, y, c=data[variable], cmap='coolwarm', marker='o', edgecolor='none', s=10, alpha=0.75, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02)
    ax.set_title(title)


fig, ax = plt.subplots(figsize=(12, 6))
    # fig, axs = plt.subplots(figsize=(10, 6), dpi=300)  # Correct subplot structure
# draw_map_subplot(ax, "HW Count Per Year(Global Summer)", 
#                  location_hw_count[location_hw_count['count'] > 60], 'count')
draw_map_subplot(ax, "HW Count Per Year(Global Summer)", 
                 location_hw_count, 'count')

plt.show()

# %% [markdown]
# # Conclusion: will use 60 days as a filtering cut in model training.


