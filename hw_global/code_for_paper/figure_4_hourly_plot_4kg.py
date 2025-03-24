import pandas as pd
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.spatial import cKDTree

THRESHOLD: int = 98
SUMMARY_DIR = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'
FIGURE_OUTPUT_DIR = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/figures_for_paper/'
MERGED_FEATHER_PATH = os.path.join(SUMMARY_DIR, f'updated_local_hour_adjusted_variables_HW{THRESHOLD}.feather')
KOPPEN_GEIGER_DATA_PATH = '/home/jguo/other_projects/1991_2020/koppen_geiger_0p5.nc'
KOPPEN_GEIGER_LEGEND_PATH = '/home/jguo/research/hw_global/Data/KoppenGeigerLegend.xlsx'
OUTPUT_FEATHER_PATH = os.path.join(SUMMARY_DIR, 'twoKG_local_hour_adjusted_variables_HW98.feather')


def replace_cold_with_continental(kg_main_group: str) -> str:
    """Replaces 'Cold' with 'Continental' in the main group name."""
    return 'Continental' if kg_main_group == 'Cold' else kg_main_group


def load_data(feather_path: str) -> pd.DataFrame:
    """Loads the main data from a feather file."""
    return pd.read_feather(feather_path)


def load_koppen_geiger_data() -> tuple[xr.Dataset, pd.DataFrame]:
    """Loads Koppen-Geiger data and legend."""
    ds_koppen_map = xr.open_dataset(KOPPEN_GEIGER_DATA_PATH)
    # Normalize longitude from -180/180 to 0/360 range
    ds_koppen_map = ds_koppen_map.assign_coords(lon=(((ds_koppen_map.lon + 360) % 360)))
    ds_koppen_map = ds_koppen_map.sortby('lon')  # Sort longitudes to maintain order
    kg_legend = pd.read_excel(KOPPEN_GEIGER_LEGEND_PATH, engine='openpyxl')
    return ds_koppen_map, kg_legend


def preprocess_koppen_geiger(ds_koppen_map: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesses the Koppen-Geiger dataset for KDTree."""
    latitudes = ds_koppen_map['lat'].values
    longitudes = ds_koppen_map['lon'].values
    lat_flat = np.repeat(latitudes, len(longitudes))
    lon_flat = np.tile(longitudes, len(latitudes))
    kg_class_flat = ds_koppen_map['kg_class'].values.flatten()

    non_zero_indices = kg_class_flat > 0
    lat_flat_non_zero = lat_flat[non_zero_indices]
    lon_flat_non_zero = lon_flat[non_zero_indices]
    kg_class_flat_non_zero = kg_class_flat[non_zero_indices]

    # Return only coordinates and KG classes
    return np.stack([lat_flat_non_zero, lon_flat_non_zero], axis=-1), kg_class_flat_non_zero


def find_nearest_non_zero_kg_class(coords: np.ndarray, tree: cKDTree, kg_class_flat_non_zero: np.ndarray) -> np.ndarray:
    """Finds the nearest non-zero Koppen-Geiger class ID using KDTree."""
    _, indices = tree.query(coords, k=1)
    return kg_class_flat_non_zero[indices]

def create_kg_mappings(kg_legend: pd.DataFrame) -> tuple[dict, dict, list]:
    """Creates mappings for Koppen-Geiger groups and a sorted list of main groups."""
    kg_legend['KGMainGroup'] = kg_legend['KGClass'].apply(lambda x: x.split(',')[0].strip())
    kg_main_group_map = dict(zip(kg_legend['ID'], kg_legend['KGMainGroup']))
    kg_major_group_map = dict(zip(kg_legend['ID'], kg_legend['KGMajorClass']))

    main_group_min_id = {}
    for kg_id, main_group in kg_main_group_map.items():
        main_group_min_id.setdefault(main_group, kg_id)
        main_group_min_id[main_group] = min(main_group_min_id[main_group], kg_id)

    sorted_main_groups = sorted(set(kg_main_group_map.values()), key=lambda x: main_group_min_id[x])
    sorted_main_groups = [group for group in sorted_main_groups if group.lower() != 'polar']
    return kg_main_group_map, kg_major_group_map, sorted_main_groups

def prepare_data_for_plotting(local_hour_adjusted_df: pd.DataFrame, kg_main_group_map: dict, kg_major_group_map: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepares the data: adds KG_ID and climate zone groups, then groups for plotting.

    Returns:
        avg_diff_by_hour_and_main_group: Aggregated UHI_diff by KGMainGroup.
        avg_diff_by_hour_and_major_class: Aggregated UHI_diff by KGMajorClass.
        local_hour_adjusted_df: The updated DataFrame.
    """
    # 1. Add KG_ID
    ds_koppen_map, kg_legend = load_koppen_geiger_data()
    coords, kg_class_flat_non_zero = preprocess_koppen_geiger(ds_koppen_map)
    tree = cKDTree(coords)

    # Prepare coordinates from the main DataFrame
    input_coords = local_hour_adjusted_df[['lat', 'lon']].to_numpy()

    local_hour_adjusted_df['KG_ID'] = find_nearest_non_zero_kg_class(input_coords, tree, kg_class_flat_non_zero)

    # 2. Add KGMainGroup using the *main* group map (for plotting)
    local_hour_adjusted_df['KGMainGroup'] = local_hour_adjusted_df['KG_ID'].map(kg_main_group_map)

    # Add KGMainGroup using the major group map (for comparison)
    local_hour_adjusted_df['KGMainGroup_Major'] = local_hour_adjusted_df['KGClass'].map(kg_major_group_map)

    # 3. Group for plotting

    # Aggregation based on KGMainGroup
    var_diff_by_localhour = local_hour_adjusted_df.groupby(['lat', 'lon', 'local_hour', 'KGMainGroup'])[['UHI_diff']].mean().reset_index().sort_values(by=['lat', 'lon', 'local_hour'])
    avg_diff_by_hour_and_main_group = var_diff_by_localhour.groupby(['KGMainGroup', 'local_hour'])[['UHI_diff']].agg(['mean', 'std']).reset_index()

    # Aggregation based on KGMajorClass (assumed to be in the DataFrame)
    var_diff_by_localhour_major = local_hour_adjusted_df.groupby(['lat', 'lon', 'local_hour', 'KGMajorClass'])[['UHI_diff']].mean().reset_index().sort_values(by=['lat', 'lon', 'local_hour'])
    avg_diff_by_hour_and_major_class = var_diff_by_localhour_major.groupby(['KGMajorClass', 'local_hour'])[['UHI_diff']].agg(['mean', 'std']).reset_index()

    return avg_diff_by_hour_and_main_group, avg_diff_by_hour_and_major_class, local_hour_adjusted_df


def get_kg_color(kg_main_group: str) -> str:
    """Returns the color for a given Koppen-Geiger main group."""
    kg_main_group_colors = {'Tropical': '#0078ff', 'Arid': '#ff9696', 'Temperate': '#96ff96', 'Cold': '#ff00ff', 'Continental':'#ff00ff'}
    return kg_main_group_colors.get(kg_main_group, '#000000')



def plot_uhi_diff(data: pd.DataFrame, group_col: str, sorted_groups: list, output_filename: str, combined: bool = False):
    """
    Generic function to plot UHI differences.

    Args:
        data: DataFrame with aggregated UHI data.
        group_col: Column name for the climate zone grouping ('KGMainGroup' or 'KGMajorClass').
        sorted_groups: List of sorted climate groups.
        output_filename: Base name for the output file.
        combined: Whether to create a combined plot (True) or individual plots (False).
    """
    min_uhi_diff = -0.75
    max_uhi_diff = 1

    if combined:
        plt.figure(figsize=(10, 6))
        for group in sorted_groups:
            subset = data[data[group_col] == group]
            group_label = replace_cold_with_continental(group)
            plt.plot(subset['local_hour'], subset[('UHI_diff', 'mean')], marker='o',
                     color=get_kg_color(group), label=group_label)

        plt.xlabel('Local Hour')
        plt.ylabel('HW-NHW UHI (°C)')
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.xticks(range(0, 24))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, f'{output_filename}_combined.png'), dpi=600, bbox_inches='tight')
        plt.close()

    else:
        n_rows = (len(sorted_groups) + 3) // 4
        fig, axs = plt.subplots(n_rows, 4, figsize=(20, 6 * n_rows), squeeze=False)

        for i, group in enumerate(sorted_groups):
            row, col = i // 4, i % 4
            subset = data[data[group_col] == group]

            axs[row, col].plot(subset['local_hour'], subset[('UHI_diff', 'mean')], marker='o',
                               color=get_kg_color(group), label='HW-NHW UHI')
            axs[row, col].fill_between(subset['local_hour'], subset[('UHI_diff', 'mean')] - subset[('UHI_diff', 'std')],
                                       subset[('UHI_diff', 'mean')] + subset[('UHI_diff', 'std')],
                                       color=get_kg_color(group), alpha=0.2, label='±1 Std Dev')

            title_font = {'size': 14, 'weight': 'normal'}
            group_font = {'size': 16, 'weight': 'bold'}
            group_label = replace_cold_with_continental(group) if group_col == 'KGMajorClass' else group
            axs[row, col].text(0.5, 1.05, group_label, fontdict=group_font,
                               transform=axs[row, col].transAxes, ha='center', va='bottom')
            axs[row, col].set_xlabel('Local Hour')
            axs[row, col].set_ylabel('HW-NHW UHI (°C)')
            axs[row, col].grid(True, axis='y')
            axs[row, col].legend()
            axs[row, col].set_ylim(min_uhi_diff, max_uhi_diff)

        for i in range(len(sorted_groups), n_rows * 4):
            fig.delaxes(axs[i // 4, i % 4])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, f'{output_filename}.png'), dpi=600, bbox_inches='tight')
        plt.close()


def compare_kg_major_and_main(local_hour_adjusted_df: pd.DataFrame):
    """Compares KGMajorClass and KGMainGroup (using the _Major suffix) in the dataframe."""

    # Use the KGMainGroup_Major column for the comparison
    location_counts = local_hour_adjusted_df.groupby('location_ID')[['KGMajorClass', 'KGMainGroup_Major']].nunique()
    inconsistent_locations_major = location_counts[location_counts['KGMajorClass'] > 1]
    inconsistent_locations_main = location_counts[location_counts['KGMainGroup_Major'] > 1]

    print("\n--- KGMajorClass and KGMainGroup Comparison ---")
    print("KGMajorClass is consistent for each location_ID." if inconsistent_locations_major.empty else
          "Inconsistencies found in KGMajorClass:\n" + str(inconsistent_locations_major))
    print("KGMainGroup_Major is consistent for each location_ID." if inconsistent_locations_main.empty else
          "Inconsistencies found in KGMainGroup_Major:\n" + str(inconsistent_locations_main))

    comparison_table = pd.crosstab(local_hour_adjusted_df['KGMajorClass'], local_hour_adjusted_df['KGMainGroup_Major'])
    print("\n--- Crosstabulation of KGMajorClass vs. KGMainGroup_Major ---")
    print(comparison_table)

    local_hour_adjusted_df['Major_Main_Match'] = local_hour_adjusted_df['KGMajorClass'] == local_hour_adjusted_df['KGMainGroup_Major']
    match_counts = local_hour_adjusted_df['Major_Main_Match'].value_counts()
    print("\n--- KGMajorClass and KGMainGroup_Major Match Summary ---", match_counts)

    non_matching_rows = local_hour_adjusted_df[~local_hour_adjusted_df['Major_Main_Match']]
    print("\n--- Rows where KGMajorClass and KGMainGroup_Major DO NOT match ---")
    if not non_matching_rows.empty:
        print(f"Number of non-matching rows: {len(non_matching_rows)}")
        print("\nSample of non-matching rows:")
        print(non_matching_rows[['location_ID', 'KGClass', 'KGMajorClass', 'KGMainGroup_Major']].sample(min(10, len(non_matching_rows))))
        print("\nValue Counts of KGClass in non-matching rows:", non_matching_rows['KGClass'].value_counts())
    else:
        print("All rows have matching KGMajorClass and KGMainGroup_Major.")
    return non_matching_rows

def normalize_longitude(lons: np.ndarray) -> np.ndarray:
    """Normalizes longitudes to the range [0, 360)."""
    return lons % 360

def save_plot(plt_obj, filename, output_dir):
    """Saves the given plot to a file."""
    filepath = os.path.join(output_dir, filename)
    plt_obj.savefig(filepath, dpi=600, bbox_inches='tight')
    plt_obj.close()


def plot_mismatched_locations(df: pd.DataFrame, output_dir: str):
    """Plots locations where KGMajorClass and KGMainGroup differ on a global map."""

    if df.empty:
        print("No mismatched KGMajorClass and KGMainGroup data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    m = Basemap(projection='cyl', lon_0=0, ax=ax, fix_aspect=False,
                llcrnrlat=-44.94133, urcrnrlat=65.12386)  # Fixed lat limits
    m.drawcoastlines(color='0.15', linewidth=0.5, zorder=3)
    m.drawcountries(linewidth=0.1)
    m.fillcontinents(color='white', lake_color='lightcyan')
    m.drawmapboundary(fill_color='lightcyan')
    m.drawparallels(np.arange(-90., 91., 30.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[0,0,0,1], fontsize=10)

    # Normalize longitudes and plot
    normalized_lons = normalize_longitude(df['lon'].values)
    x, y = m(normalized_lons, df['lat'].values)
    m.scatter(x, y, color='red', marker='o', s=10, alpha=0.75, zorder=4)

    ax.set_title('Locations with Mismatched KGMajorClass and KGMainGroup')

    save_plot(plt, "mismatched_kg_locations.png", output_dir)



def main():
    """Main function to execute the analysis and plotting."""

    hourly_plot_4kg_feather = FIGURE_OUTPUT_DIR + 'Figure_3_hourly_Plot_4KG.feather'
    if os.path.exists(hourly_plot_4kg_feather):
        print(f"Loading data from {hourly_plot_4kg_feather}")

        avg_diff_by_hour_and_major_class = pd.read_feather(hourly_plot_4kg_feather)
    else:
        print(f"File {hourly_plot_4kg_feather} does not exist. Running the analysis...")    
        local_hour_adjusted_df = load_data(MERGED_FEATHER_PATH)
        ds_koppen_map, kg_legend = load_koppen_geiger_data()
        kg_main_group_map, kg_major_group_map, sorted_main_groups = create_kg_mappings(kg_legend)

        # Prepare data and add KG_ID *before* comparison, get both aggregated DataFrames and the updated DataFrame
        avg_diff_by_hour_and_main_group, avg_diff_by_hour_and_major_class, local_hour_adjusted_df = prepare_data_for_plotting(local_hour_adjusted_df, kg_main_group_map, kg_major_group_map)

        # Now do the comparison and get the non-matching rows
        # non_matching_rows = compare_kg_major_and_main(local_hour_adjusted_df)

        # Save the updated DataFrame *after* adding KGMainGroup and KG_ID
        # local_hour_adjusted_df.to_feather(OUTPUT_FEATHER_PATH)
        # print(f"Saved updated DataFrame to {OUTPUT_FEATHER_PATH}")

        # # Use the generic plotting function
        # plot_uhi_diff(avg_diff_by_hour_and_main_group, 'KGMainGroup', sorted_main_groups, 'kg_main_group_uhi_diff')
        avg_diff_by_hour_and_major_class.to_feather(hourly_plot_4kg_feather)
    
    sorted_major_classes = sorted(avg_diff_by_hour_and_major_class['KGMajorClass'].unique())
    # remove polar from sorted_major_classes
    sorted_major_classes = [cls for cls in sorted_major_classes if cls != 'Polar']
    plot_uhi_diff(avg_diff_by_hour_and_major_class, 'KGMajorClass', sorted_major_classes, 'Figure_4_hourly_Plot_4KG')
    # plot_uhi_diff(avg_diff_by_hour_and_major_class, 'KGMajorClass', sorted_major_classes, 'kg_major_class_uhi_diff', combined=True)
    # Plot mismatched locations
    # plot_mismatched_locations(non_matching_rows, FIGURE_OUTPUT_DIR)

if __name__ == "__main__":
    main()