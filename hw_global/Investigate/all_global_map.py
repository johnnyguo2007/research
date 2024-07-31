import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import argparse
from scipy.interpolate import griddata

def normalize_longitude(lons):
    return np.where(lons > 180, lons - 360, lons)

def draw_map_scatter(data, variable, output_file):
    fig, ax = plt.subplots(figsize=(12, 6))
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

    lons = normalize_longitude(data['lon'].values)
    x, y = m(lons, data['lat'].values)
    
    sc = m.scatter(x, y, c=data[variable], cmap='RdBu_r', 
                   s=10, alpha=0.7, edgecolors='none', zorder=4)
    
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, extend='both')
    cbar.set_label(variable)

    ax.set_title(variable)

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot for {variable} saved as {output_file}")

def draw_map_pcolormesh(data, variable, output_file):
    fig, ax = plt.subplots(figsize=(12, 6))
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

    lons = normalize_longitude(data['lon'].values)
    lats = data['lat'].values

    lon_grid, lat_grid = np.mgrid[-180:180:360j, -90:90:180j]
    grid_data = griddata((lons, lats), data[variable].values, (lon_grid, lat_grid), method='linear')
    masked_data = np.ma.masked_invalid(grid_data)

    lon_edges = np.linspace(-180, 180, 361)
    lat_edges = np.linspace(-90, 90, 181)

    sc = m.pcolormesh(lon_edges, lat_edges, masked_data.T, cmap='RdBu_r', latlon=True, shading='flat')
    
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, extend='both')
    cbar.set_label(variable)

    ax.set_title(variable)

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot for {variable} saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate or load global map data.')
    parser.add_argument('--generate', action='store_true', help='Generate the output feather files.')
    parser.add_argument('--plot-type', choices=['scatter', 'pcolormesh'], default='scatter', help='Choose the plot type: scatter or pcolormesh')
    args = parser.parse_args()

    output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/plots'
    os.makedirs(output_dir, exist_ok=True)

    if args.generate:
        df = pd.read_feather('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/updated_local_hour_adjusted_variables_HW98.feather')
        exclude_columns = ['HW98', 'location_ID', 'event_ID', 'global_event_ID', 'hour', 'month', 'year', 'local_time', 'local_hour', 'TOPO', 'KGClass', 'KGMajorClass', 'time']
        numeric_columns = df.select_dtypes(include=[np.number]).columns.difference(exclude_columns + ['lat', 'lon']).tolist()
        df_grouped = df.groupby(['lat', 'lon'])[numeric_columns].mean().reset_index()
        df_grouped['lon'] = normalize_longitude(df_grouped['lon'])
        df_grouped.to_feather(os.path.join(output_dir, 'grouped_global_map.feather'))
    else:
        df_grouped = pd.read_feather(os.path.join(output_dir, 'grouped_global_map.feather'))

    variables = [col for col in df_grouped.columns if col not in ['lat', 'lon']]

    draw_function = draw_map_scatter if args.plot_type == 'scatter' else draw_map_pcolormesh

    for variable in variables:
        try:
            if not np.issubdtype(df_grouped[variable].dtype, np.number):
                print(f"Skipping {variable} as it's not a numeric type.")
                continue

            output_file = os.path.join(output_dir, f'{variable}_global_map_{args.plot_type}.png')
            draw_function(df_grouped, variable, output_file)
        except Exception as e:
            print(f"Error plotting {variable}: {str(e)}")

    print(f"All plots have been saved in the directory: {output_dir}")

if __name__ == "__main__":
    main()