import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import argparse
from matplotlib.colors import LinearSegmentedColormap, Normalize

def normalize_longitude(lons):
    return np.where(lons > 180, lons - 360, lons)

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

    return fig, ax, m

def draw_map_pcolormesh(data, variable, output_file, time_of_day):
    print(f"Processing {variable}")
    print(f"Data shape: {data[variable].shape}")
    print(f"Data type: {data[variable].dtype}")
    print(f"Sample data: {data[variable].head()}")

    fig, ax, m = setup_map()

    lons = normalize_longitude(data['lon'].values)
    lats = data['lat'].values
    values = data[variable].values

    print(f"Debug - {variable} shape: {values.shape}, min: {np.nanmin(values)}, max: {np.nanmax(values)}")

    # Set colormap based on variable name
    if 'diff' in variable.lower() or 'delta' in variable.lower() or 'uhi' in variable.lower():
        cmap = 'RdBu_r'  # Default colormap
        # Determine the maximum absolute value for symmetric color scaling
        max_abs_val = max(abs(np.nanmin(values)), abs(np.nanmax(values))) * 0.6
        norm = plt.Normalize(-max_abs_val, max_abs_val)
    else:
        cmap = 'RdBu_r'  # Default colormap
        norm = None

    lon_grid, lat_grid = np.meshgrid(np.unique(lons), np.unique(lats))
    value_grid = np.full(lon_grid.shape, np.nan)

    for i in range(len(lats)):
        lat_idx = np.where(lat_grid[:,0] == lats[i])[0][0]
        lon_idx = np.where(lon_grid[0,:] == lons[i])[0][0]
        value_grid[lat_idx, lon_idx] = values[i]

    sc = m.pcolormesh(lon_grid, lat_grid, value_grid, cmap=cmap, norm=norm, latlon=True)

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, extend='both')
    cbar.set_label(variable)

    ax.set_title(f"{variable} - {time_of_day.capitalize()}")

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Pcolormesh plot for {variable} ({time_of_day}) saved as {output_file}")

def draw_map_scatter(data, variable, output_file, time_of_day):
    fig, ax, m = setup_map()

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

    ax.set_title(f"{variable} - {time_of_day.capitalize()}")

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Scatter plot for {variable} ({time_of_day}) saved as {output_file}")

def create_new_variables(df):
    df['total_available_energy'] = df['FSA'] + df['FIRA'] - df['FGR']
    df['total_available_energy_U'] = df['FSA_U'] + df['FIRA_U'] - df['FGR_U']
    df['total_available_energy_R'] = df['FSA_R'] + df['FIRA_R'] - df['FGR_R']

    df['Delta_total_available_energy'] = df['total_available_energy_U'] - df['total_available_energy_R']

    df['hw_nohw_diff_total_available_energy_U'] = df['hw_nohw_diff_FSA_U'] + df['hw_nohw_diff_FIRA_U'] - df['hw_nohw_diff_FGR_U']
    df['hw_nohw_diff_total_available_energy_R'] = df['hw_nohw_diff_FSA_R'] + df['hw_nohw_diff_FIRA_R'] - df['hw_nohw_diff_FGR_R']
    df['hw_nohw_diff_total_available_energy'] = df['hw_nohw_diff_FSA'] + df['hw_nohw_diff_FIRA'] - df['hw_nohw_diff_FGR']

    df['Double_Differencing_total_available_energy'] = df['hw_nohw_diff_total_available_energy_U'] - df['hw_nohw_diff_total_available_energy_R']

    for column in df.columns:
        if not column.startswith('hw_nohw_diff') and column.endswith('_U'):
            base_var = column[:-2]  # Remove '_U' from the end
            if f'{base_var}_R' in df.columns:
                df[f'Delta_{base_var}'] = df[f'{base_var}_U'] - df[f'{base_var}_R']

        if column.startswith('hw_nohw_diff') and column.endswith('_U'):
            base_var = column[len('hw_nohw_diff_'):-2]  # Remove 'hw_nohw_diff_' from start and '_U' from end
            if f'hw_nohw_diff_{base_var}_R' in df.columns:
                df[f'Double_Differencing_{base_var}'] = df[column] - df[f'hw_nohw_diff_{base_var}_R']

    return df

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Generate or load global map data.')
    parser.add_argument('--generate', action='store_true', help='Generate the output feather files.')
    parser.add_argument('--plot-type', choices=['pcolormesh', 'scatter'], default='pcolormesh', help='Choose the plotting method')
    parser.add_argument('--variables', type=str, help='Comma-separated list of variables to plot')
    args = parser.parse_args()

    # Create a directory to save the plots and DataFrames
    output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/new_plots'
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

    # Create new variables
    df_grouped_daytime = create_new_variables(df_grouped_daytime)
    df_grouped_nighttime = create_new_variables(df_grouped_nighttime)

    # Only plot the new total_available_energy variables
    variables = [
        'total_available_energy',
        'total_available_energy_U',
        'total_available_energy_R',
        'Delta_total_available_energy',
        'hw_nohw_diff_total_available_energy',
        'Double_Differencing_total_available_energy'
    ]

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
            plot_function(df_grouped_daytime, variable, output_file, "day")

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
            plot_function(df_grouped_nighttime, variable, output_file, "night")

        except Exception as e:
            print(f"Error plotting {variable}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"All daytime plots have been saved in the directory: {output_dir_daytime}")
    print(f"All nighttime plots have been saved in the directory: {output_dir_nighttime}")

if __name__ == "__main__":
    main()