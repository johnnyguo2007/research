import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import argparse
from scipy.interpolate import griddata

def create_land_sea_mask(m, lons, lats):
    print(f"Debug - create_land_sea_mask input shapes: lons {lons.shape}, lats {lats.shape}")
    xx, yy = np.meshgrid(lons, lats)
    mask = np.zeros(xx.shape, dtype=bool)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            mask[i, j] = m.is_land(xx[i, j], yy[i, j])
    return mask

def draw_map_pcolormesh(data, variable, output_file):
    print(f"Processing {variable}")
    print(f"Data shape: {data[variable].shape}")
    print(f"Data type: {data[variable].dtype}")
    print(f"Sample data: {data[variable].head()}")

    fig, ax = plt.subplots(figsize=(12, 6))
    m = Basemap(projection='cyl', lon_0=0, ax=ax,
                fix_aspect=False,
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180)
    m.drawcoastlines(color='0.15', linewidth=0.5)
    m.drawcountries(linewidth=0.1)
    m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1], fontsize=10)

    # Create a regular grid for pcolormesh
    lon_grid, lat_grid = np.meshgrid(np.linspace(-180, 180, 360), np.linspace(-90, 90, 180))

    # Interpolate data onto the regular grid
    values = griddata((data['lon'], data['lat']), data[variable], (lon_grid, lat_grid), method='linear')

    print(f"Debug - {variable} shape: {values.shape}, min: {np.nanmin(values)}, max: {np.nanmax(values)}")

    # Create land-sea mask
    mask = create_land_sea_mask(m, lon_grid[0, :], lat_grid[:, 0])

    # Apply the mask
    masked_values = np.ma.array(values, mask=~mask)

    # Plot using pcolormesh
    sc = m.pcolormesh(lon_grid, lat_grid, masked_values, cmap='RdBu_r', latlon=True)

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, extend='both')
    cbar.set_label(variable)

    ax.set_title(variable)

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot for {variable} saved as {output_file}")

def draw_map_contourf(data, variable, output_file):
    fig, ax = plt.subplots(figsize=(12, 6))
    m = Basemap(projection='cyl', lon_0=0, ax=ax,
                fix_aspect=False,
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180)
    m.drawcoastlines(color='0.15', linewidth=0.5)
    m.drawcountries(linewidth=0.1)
    m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1], fontsize=10)

    # Interpolate data to create a filled contour plot
    lat_range = np.linspace(data['lat'].min(), data['lat'].max(), 180)
    lon_range = np.linspace(data['lon'].min(), data['lon'].max(), 360)
    lons, lats = np.meshgrid(lon_range, lat_range)
    values = griddata((data['lat'], data['lon']), data[variable], (lats, lons), method='linear')

    # Plot using contourf
    sc = m.contourf(lons, lats, values, cmap='RdBu_r', latlon=True)

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, extend='both')
    cbar.set_label(variable)

    ax.set_title(variable)

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot for {variable} saved as {output_file}")

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Generate or load global map data.')
    parser.add_argument('--generate', action='store_true', help='Generate the output feather files.')
    parser.add_argument('--plot-type', choices=['pcolormesh', 'contourf'], default='pcolormesh', help='Choose the plotting method')
    args = parser.parse_args()

    # Create a directory to save the plots and DataFrames
    output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/plots'
    os.makedirs(output_dir, exist_ok=True)

    if args.generate:
        # Load the data
        df = pd.read_feather('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/updated_local_hour_adjusted_variables_HW98.feather')

        # Columns to exclude from averaging and plotting
        exclude_columns = ['HW98', 'location_ID', 'event_ID', 'global_event_ID', 'hour', 'month', 'year', 'local_time', 'local_hour', 'TOPO', 'KGClass', 'KGMajorClass', 'time', 'lat', 'lon']
        
        # Get list of numeric columns excluding the ones to be removed
        numeric_columns = df.select_dtypes(include=[np.number]).columns.difference(exclude_columns).tolist()

        # Group by lat and lon, and average all numeric columns
        df_grouped = df.groupby(['lat', 'lon'])[numeric_columns].mean()

        def normalize_longitude(lons):
            return np.where(lons > 180, lons - 360, lons)

        # Normalize longitude values in the DataFrame
        df_grouped = df_grouped.reset_index()
        df_grouped['lon'] = normalize_longitude(df_grouped['lon'])

        # Save the resulting DataFrame
        df_grouped.to_feather(os.path.join(output_dir, 'grouped_global_map.feather'))
    else:
        # Load the grouped_global_map DataFrame from feather files
        df_grouped = pd.read_feather(os.path.join(output_dir, 'grouped_global_map.feather'))

    # Choose the drawing function based on the plot type
    draw_function = draw_map_pcolormesh if args.plot_type == 'pcolormesh' else draw_map_contourf

    # Get the list of variables to plot
    variables = [col for col in df_grouped.columns if col not in ['lat', 'lon']]

    # Plot each variable in a separate PNG file
    for variable in variables:
        try:
            # Check if the column is numeric
            if not np.issubdtype(df_grouped[variable].dtype, np.number):
                print(f"Skipping {variable} as it's not a numeric type.")
                continue

            output_file = os.path.join(output_dir, f'{variable}_global_map.png')
            draw_function(df_grouped, variable, output_file)
        except Exception as e:
            print(f"Error plotting {variable}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"All plots have been saved in the directory: {output_dir}")

if __name__ == "__main__":
    main()