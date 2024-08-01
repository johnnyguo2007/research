import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import argparse
from scipy.interpolate import griddata


def normalize_longitude(lons):
    return np.where(lons > 180, lons - 360, lons)

def create_land_sea_mask(m, lons, lats):
    print(f"Debug - create_land_sea_mask input shapes: lons {lons.shape}, lats {lats.shape}")
    xx, yy = np.meshgrid(lons, lats)
    mask = np.zeros(xx.shape, dtype=bool)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            mask[i, j] = m.is_land(xx[i, j], yy[i, j])
    return mask

def setup_map():
    fig, ax = plt.subplots(figsize=(12, 6))
    m = Basemap(projection='cyl', lon_0=0, ax=ax,
                fix_aspect=False,
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180)
    m.drawcoastlines(color='0.15', linewidth=0.5)
    m.fillcontinents(color='white', lake_color='lightcyan')
    m.drawmapboundary(fill_color='lightcyan')
    m.drawcountries(linewidth=0.1)
    m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1], fontsize=10)

    # Create a regular grid for pcolormesh and mask calculation
    lon_grid, lat_grid = np.meshgrid(np.linspace(-180, 180, 360), np.linspace(-90, 90, 180))
    mask = create_land_sea_mask(m, lon_grid[0, :], lat_grid[:, 0])

    return fig, ax, m, lon_grid, lat_grid, mask

def draw_map_pcolormesh(data, variable, output_file, lon_grid, lat_grid, mask):
    print(f"Processing {variable}")
    print(f"Data shape: {data[variable].shape}")
    print(f"Data type: {data[variable].dtype}")
    print(f"Sample data: {data[variable].head()}")

    fig, ax, m, _, _, _ = setup_map()

    # Interpolate data onto the regular grid
    values = griddata((data['lon'], data['lat']), data[variable], (lon_grid, lat_grid), method='linear')

    print(f"Debug - {variable} shape: {values.shape}, min: {np.nanmin(values)}, max: {np.nanmax(values)}")

    # Apply the mask
    masked_values = np.ma.array(values, mask=~mask)

    # Set colormap based on variable name
    cmap = 'RdBu_r'  # Default colormap
    if 'diff' in variable or 'delta' in variable:
        cmap = 'RdBu'  # Centered colormap for difference variables

    sc = m.pcolormesh(lon_grid, lat_grid, masked_values, cmap=cmap, latlon=True)

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, extend='both')
    cbar.set_label(variable)

    ax.set_title(variable)

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Pcolormesh plot for {variable} saved as {output_file}")

def draw_map_scatter(data, variable, output_file, mask):
    fig, ax, m, _, _, _ = setup_map()

    lons = normalize_longitude(data['lon'].values)
    x, y = m(lons, data['lat'].values)

    # Set colormap based on variable name
    cmap = 'RdBu_r'  # Default colormap
    if 'diff' in variable or 'delta' in variable:
        cmap = 'RdBu'  # Centered colormap for difference variables

    sc = m.scatter(x, y, c=data[variable], cmap=cmap,
                   s=10, alpha=0.7, edgecolors='none', zorder=4)

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, extend='both')
    cbar.set_label(variable)

    ax.set_title(variable)

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Scatter plot for {variable} saved as {output_file}")


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Generate or load global map data.')
    parser.add_argument('--generate', action='store_true', help='Generate the output feather files.')
    parser.add_argument('--plot-type', choices=['pcolormesh', 'scatter'], default='pcolormesh', help='Choose the plotting method')
    args = parser.parse_args()

    # Create a directory to save the plots and DataFrames
    output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/plots'
    os.makedirs(output_dir, exist_ok=True)

    # Define output directories for daytime and nighttime plots
    output_dir_daytime = os.path.join(output_dir, 'day')
    output_dir_nighttime = os.path.join(output_dir, 'night')
    os.makedirs(output_dir_daytime, exist_ok=True)
    os.makedirs(output_dir_nighttime, exist_ok=True)

    if args.generate:
        # Load the data
        df = pd.read_feather('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/updated_local_hour_adjusted_variables_HW98.feather')

        # Columns to exclude from averaging and plotting
        exclude_columns = ['HW98', 'location_ID', 'event_ID', 'global_event_ID', 'hour', 'month', 'year', 'local_time', 'local_hour', 'TOPO', 'KGClass', 'KGMajorClass', 'time', 'lat', 'lon']

        # Get list of numeric columns excluding the ones to be removed
        numeric_columns = df.select_dtypes(include=[np.number]).columns.difference(exclude_columns).tolist()

        # Create separate DataFrames for daytime and nighttime data
        daytime_mask = df['local_hour'].between(7, 16)
        nighttime_mask = (df['local_hour'].between(20, 24) | df['local_hour'].between(0, 6))

        df_daytime = df[daytime_mask]
        df_nighttime = df[nighttime_mask]

        # Process daytime data
        df_grouped_daytime = df_daytime.groupby(['lat', 'lon'])[numeric_columns].mean()
        df_grouped_daytime = df_grouped_daytime.reset_index()
        df_grouped_daytime['lon'] = normalize_longitude(df_grouped_daytime['lon'])
        df_grouped_daytime.to_feather(os.path.join(output_dir_daytime, 'grouped_global_map_daytime.feather'))

        # Process nighttime data
        df_grouped_nighttime = df_nighttime.groupby(['lat', 'lon'])[numeric_columns].mean()
        df_grouped_nighttime = df_grouped_nighttime.reset_index()
        df_grouped_nighttime['lon'] = normalize_longitude(df_grouped_nighttime['lon'])
        df_grouped_nighttime.to_feather(os.path.join(output_dir_nighttime, 'grouped_global_map_nighttime.feather'))
    else:
        # Load the grouped_global_map DataFrames from feather files for daytime and nighttime
        df_grouped_daytime = pd.read_feather(os.path.join(output_dir_daytime, 'grouped_global_map_daytime.feather'))
        df_grouped_nighttime = pd.read_feather(os.path.join(output_dir_nighttime, 'grouped_global_map_nighttime.feather'))

    # Get the list of variables to plot
    variables = [col for col in df_grouped_daytime.columns if col not in ['lat', 'lon']]

    # Calculate the land-sea mask and regular grid once
    _, _, _, lon_grid, lat_grid, mask = setup_map()

    # Choose the plotting function based on the --plot-type argument
    if args.plot_type == 'pcolormesh':
        plot_function = draw_map_pcolormesh
    elif args.plot_type == 'scatter':
        plot_function = draw_map_scatter
    else:
        raise ValueError(f"Invalid plot type: {args.plot_type}")

    # Plot daytime data
    for variable in variables:
        try:
            # Check if the column is numeric
            if not np.issubdtype(df_grouped_daytime[variable].dtype, np.number):
                print(f"Skipping {variable} as it's not a numeric type.")
                continue

            output_file = os.path.join(output_dir_daytime, f'{variable}_global_map_daytime_{args.plot_type}.png')
            plot_function(df_grouped_daytime, variable, output_file, lon_grid, lat_grid, mask)

        except Exception as e:
            print(f"Error plotting {variable}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Plot nighttime data
    for variable in variables:
        try:
            # Check if the column is numeric
            if not np.issubdtype(df_grouped_nighttime[variable].dtype, np.number):
                print(f"Skipping {variable} as it's not a numeric type.")
                continue

            output_file = os.path.join(output_dir_nighttime, f'{variable}_global_map_nighttime_{args.plot_type}.png')
            plot_function(df_grouped_nighttime, variable, output_file, lon_grid, lat_grid, mask)

        except Exception as e:
            print(f"Error plotting {variable}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"All daytime plots have been saved in the directory: {output_dir_daytime}")
    print(f"All nighttime plots have been saved in the directory: {output_dir_nighttime}")

if __name__ == "__main__":
    main()
