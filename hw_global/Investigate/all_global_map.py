import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os

# Load the data
df = pd.read_feather('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/updated_local_hour_adjusted_variables_HW98.feather')

# Columns to exclude from averaging
exclude_columns = ['HW98', 'location_ID', 'event_ID', 'global_event_ID', 'hour', 'month', 'year', 'local_time', 'local_hour', 'TOPO', 'KGClass', 'KGMajorClass']

# Group by lat and lon, and average all other columns except the excluded ones
grouped = df.groupby(['lat', 'lon']).agg({col: 'mean' for col in df.columns if col not in exclude_columns and col not in ['lat', 'lon']}).reset_index()

def normalize_longitude(lons):
    return np.where(lons > 180, lons - 360, lons)

def draw_map_subplot(ax, title, data, variable):
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
    lons, lats = np.meshgrid(np.unique(normalized_lons), np.unique(data['lat'].values))
    
    values = data.pivot(index='lat', columns='lon', values=variable).values
    
    im = m.pcolormesh(lons, lats, values, cmap='RdBu_r', latlon=True)
    
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, extend='both')
    cbar.set_label(variable)
    
    ax.set_title(title)

# Create a directory to save the plots
os.makedirs('plots', exist_ok=True)

# Plot each variable
for variable in grouped.columns:
    if variable not in ['lat', 'lon']:
        fig, ax = plt.subplots(figsize=(12, 6))
        draw_map_subplot(ax, variable, grouped, variable)
        plt.savefig(f'/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/plots/{variable}.png', 
                    dpi=600, bbox_inches='tight')
        plt.close(fig)

print("All plots have been saved in the 'plots' directory.")