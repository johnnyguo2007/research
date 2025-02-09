import pandas as pd
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt

THRESHOLD: int = 98
SUMMARY_DIR = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'
FIGURE_OUTPUT_DIR = '/home/jguo/tmp/output2'
MERGED_FEATHER_PATH = os.path.join(SUMMARY_DIR, f'updated_local_hour_adjusted_variables_HW{THRESHOLD}.feather')
KOPPEN_GEIGER_DATA_PATH = '/home/jguo/other_projects/1991_2020/koppen_geiger_0p5.nc'
KOPPEN_GEIGER_LEGEND_PATH = '/home/jguo/research/hw_global/Data/KoppenGeigerLegend.xlsx'


def replace_cold_with_continental(kg_main_group: str) -> str:
    """Replaces 'Cold' with 'Continental' in the main group name."""
    return 'Continental' if kg_main_group == 'Cold' else kg_main_group


def load_data(feather_path: str) -> pd.DataFrame:
    """Loads the main data from a feather file."""
    return pd.read_feather(feather_path)


def load_koppen_geiger_data() -> tuple[xr.Dataset, pd.DataFrame]:
    """Loads Koppen-Geiger data and legend."""
    ds_koppen_map = xr.open_dataset(KOPPEN_GEIGER_DATA_PATH)
    kg_legend = pd.read_excel(KOPPEN_GEIGER_LEGEND_PATH, engine='openpyxl')
    return ds_koppen_map, kg_legend


def preprocess_koppen_geiger(ds_koppen_map: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocesses the Koppen-Geiger dataset for nearest neighbor search."""
    latitudes = ds_koppen_map['lat'].values
    longitudes = ds_koppen_map['lon'].values
    lat_flat = np.repeat(latitudes, len(longitudes))
    lon_flat = np.tile(longitudes, len(latitudes))
    kg_class_flat = ds_koppen_map['kg_class'].values.flatten()

    non_zero_indices = kg_class_flat > 0
    return lat_flat[non_zero_indices], lon_flat[non_zero_indices], kg_class_flat[non_zero_indices]


def find_nearest_non_zero_kg_class(lat: np.ndarray, lon: np.ndarray, lat_flat_non_zero: np.ndarray,
                                  lon_flat_non_zero: np.ndarray, kg_class_flat_non_zero: np.ndarray) -> np.ndarray:
    """Finds the nearest non-zero Koppen-Geiger class ID for multiple lat/lon pairs."""
    # Calculate distances for all lat/lon pairs at once
    distances = np.sqrt((lat_flat_non_zero - lat[:, None])**2 + (lon_flat_non_zero - lon[:, None])**2)
    # Find the index of the minimum distance for each lat/lon pair
    nearest_indices = np.argmin(distances, axis=1)
    # Return the corresponding KG class IDs
    return kg_class_flat_non_zero[nearest_indices]


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

def prepare_data_for_plotting(local_hour_adjusted_df: pd.DataFrame, kg_main_group_map: dict) -> pd.DataFrame:
    """Prepares the data for plotting by grouping and aggregating."""
    var_diff_by_localhour = local_hour_adjusted_df.groupby(['lat', 'lon', 'local_hour'])[['UHI_diff']].mean().reset_index().sort_values(by=['lat', 'lon', 'local_hour'])
    
    # Prepare Koppen-Geiger Data and find nearest KG class
    ds_koppen_map, kg_legend = load_koppen_geiger_data()
    lat_flat_non_zero, lon_flat_non_zero, kg_class_flat_non_zero = preprocess_koppen_geiger(ds_koppen_map)
    # No need to vectorize, the function now handles arrays directly
    var_diff_by_localhour['KG_ID'] = find_nearest_non_zero_kg_class(
        var_diff_by_localhour['lat'].values, var_diff_by_localhour['lon'].values,
        lat_flat_non_zero, lon_flat_non_zero, kg_class_flat_non_zero
    )

    var_diff_by_localhour['KGMainGroup'] = var_diff_by_localhour['KG_ID'].map(kg_main_group_map)
    avg_diff_by_hour_and_main_group = var_diff_by_localhour.groupby(['KGMainGroup', 'local_hour'])[['UHI_diff']].agg(['mean', 'std']).reset_index()
    return avg_diff_by_hour_and_main_group

def get_kg_color(kg_main_group: str) -> str:
    """Returns the color for a given Koppen-Geiger main group."""
    kg_main_group_colors = {'Tropical': '#0078ff', 'Arid': '#ff9696', 'Temperate': '#96ff96', 'Cold': '#ff00ff'}
    return kg_main_group_colors.get(kg_main_group, '#000000')

def plot_individual_climate_zones(avg_diff_by_hour_and_main_group: pd.DataFrame, sorted_main_groups: list):
    """Plots individual climate zone UHI differences."""
    min_uhi_diff = -0.75
    max_uhi_diff = 1
    n_rows = (len(sorted_main_groups) + 3) // 4
    fig, axs = plt.subplots(n_rows, 4, figsize=(20, 6 * n_rows), squeeze=False)

    for i, main_group in enumerate(sorted_main_groups):
        row, col = i // 4, i % 4
        subset = avg_diff_by_hour_and_main_group[avg_diff_by_hour_and_main_group['KGMainGroup'] == main_group]

        axs[row, col].plot(subset['local_hour'], subset[('UHI_diff', 'mean')], marker='o',
                           color=get_kg_color(main_group), label='HW-NHW UHI')
        axs[row, col].fill_between(subset['local_hour'], subset[('UHI_diff', 'mean')] - subset[('UHI_diff', 'std')],
                                   subset[('UHI_diff', 'mean')] + subset[('UHI_diff', 'std')],
                                   color=get_kg_color(main_group), alpha=0.2, label='±1 Std Dev')

        title_font = {'size': 14, 'weight': 'normal'}
        main_group_font = {'size': 16, 'weight': 'bold'}
        axs[row, col].set_title('Climate Zone:', fontdict=title_font)
        axs[row, col].text(0.5, 1.05, replace_cold_with_continental(main_group), fontdict=main_group_font,
                           transform=axs[row, col].transAxes, ha='center', va='bottom')
        axs[row, col].set_xlabel('Local Hour')
        axs[row, col].set_ylabel('Average UHI_diff')
        axs[row, col].grid(True, axis='y')
        axs[row, col].legend()
        axs[row, col].set_ylim(min_uhi_diff, max_uhi_diff)

    for i in range(len(sorted_main_groups), n_rows * 4):
        fig.delaxes(axs[i // 4, i % 4])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Average Hourly HW-NHW UHI by Climate Zone', size=20, weight='bold', y=0.99)
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, 'kg_main_group_uhi_diff.png'), dpi=600, bbox_inches='tight')
    plt.close()


def plot_combined_climate_zones(avg_diff_by_hour_and_main_group: pd.DataFrame, sorted_main_groups: list):
    """Plots a combined chart of all main groups without standard deviation."""
    plt.figure(figsize=(10, 6))
    for main_group in sorted_main_groups:
        subset = avg_diff_by_hour_and_main_group[avg_diff_by_hour_and_main_group['KGMainGroup'] == main_group]
        plt.plot(subset['local_hour'], subset[('UHI_diff', 'mean')], marker='o',
                 color=get_kg_color(main_group), label=replace_cold_with_continental(main_group))

    plt.title('Average Hourly UHI_diff by KG Main Groups')
    plt.xlabel('Local Hour')
    plt.ylabel('Average UHI_diff')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, 'kg_main_group_uhi_diff_combined.png'), dpi=600, bbox_inches='tight')
    plt.close()


def compare_kg_major_and_main(local_hour_adjusted_df: pd.DataFrame, kg_major_group_map : dict):
    """Compares KGMajorClass and KGMainGroup in the dataframe."""
    
    local_hour_adjusted_df['KGMainGroup'] = local_hour_adjusted_df['KGClass'].map(kg_major_group_map)
    location_counts = local_hour_adjusted_df.groupby('location_ID')[['KGMajorClass', 'KGMainGroup']].nunique()
    inconsistent_locations_major = location_counts[location_counts['KGMajorClass'] > 1]
    inconsistent_locations_main = location_counts[location_counts['KGMainGroup'] > 1]

    print("\n--- KGMajorClass and KGMainGroup Comparison ---")
    print("KGMajorClass is consistent for each location_ID." if inconsistent_locations_major.empty else
          "Inconsistencies found in KGMajorClass:\n" + str(inconsistent_locations_major))
    print("KGMainGroup is consistent for each location_ID." if inconsistent_locations_main.empty else
          "Inconsistencies found in KGMainGroup:\n" + str(inconsistent_locations_main))

    comparison_table = pd.crosstab(local_hour_adjusted_df['KGMajorClass'], local_hour_adjusted_df['KGMainGroup'])
    print("\n--- Crosstabulation of KGMajorClass vs. KGMainGroup ---")
    print(comparison_table)

    local_hour_adjusted_df['Major_Main_Match'] = local_hour_adjusted_df['KGMajorClass'] == local_hour_adjusted_df['KGMainGroup']
    match_counts = local_hour_adjusted_df['Major_Main_Match'].value_counts()
    print("\n--- KGMajorClass and KGMainGroup Match Summary ---", match_counts)

    non_matching_rows = local_hour_adjusted_df[~local_hour_adjusted_df['Major_Main_Match']]
    print("\n--- Rows where KGMajorClass and KGMainGroup DO NOT match ---")
    if not non_matching_rows.empty:
        print(f"Number of non-matching rows: {len(non_matching_rows)}")
        print("\nSample of non-matching rows:")
        print(non_matching_rows[['location_ID', 'KGClass', 'KGMajorClass', 'KGMainGroup']].sample(min(10, len(non_matching_rows))))
        print("\nValue Counts of KGClass in non-matching rows:", non_matching_rows['KGClass'].value_counts())
    else:
        print("All rows have matching KGMajorClass and KGMainGroup.")




def main():
    """Main function to execute the analysis and plotting."""
    local_hour_adjusted_df = load_data(MERGED_FEATHER_PATH)
    ds_koppen_map, kg_legend = load_koppen_geiger_data()
    kg_main_group_map, kg_major_group_map, sorted_main_groups = create_kg_mappings(kg_legend)

    avg_diff_by_hour_and_main_group = prepare_data_for_plotting(local_hour_adjusted_df, kg_main_group_map)

    plot_individual_climate_zones(avg_diff_by_hour_and_main_group, sorted_main_groups)
    plot_combined_climate_zones(avg_diff_by_hour_and_main_group, sorted_main_groups)
    compare_kg_major_and_main(local_hour_adjusted_df, kg_major_group_map)


if __name__ == "__main__":
    main()