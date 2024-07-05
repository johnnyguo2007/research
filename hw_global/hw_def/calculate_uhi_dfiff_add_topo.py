import xarray as xr
import pandas as pd
import numpy as np
import os
import argparse

def calculate_uhi_diff(df_hw, df_no_hw_avg):
    """
    Calculate the difference in UHI and UWBI between heatwave and non-heatwave periods.
    
    Args:
    df_hw (pd.DataFrame): DataFrame containing heatwave data
    df_no_hw_avg (pd.DataFrame): DataFrame containing averaged non-heatwave data
    
    Returns:
    pd.DataFrame: Merged DataFrame with UHI and UWBI differences
    """
    print("Calculating UHI and UWBI differences...")
    merged_df = pd.merge(df_hw, df_no_hw_avg[['lat', 'lon', 'year', 'hour', 'UHI', 'UWBI']], 
                         on=['lat', 'lon', 'year', 'hour'],
                         suffixes=('', '_avg'))
    merged_df['UHI_diff'] = merged_df['UHI'] - merged_df['UHI_avg']
    merged_df['UWBI_diff'] = merged_df['UWBI'] - merged_df['UWBI_avg']
    print("UHI and UWBI differences calculated.")
    return merged_df

def main(summary_dir, hw_threshold):
    """
    Main function to process HW and no_HW data and add TOPO information.
    
    Args:
    summary_dir (str): Directory containing the summary data files
    hw_threshold (int): Heatwave threshold (95 or 90)
    """
    print(f"Processing data in directory: {summary_dir}")
    print(f"Using HW threshold: {hw_threshold}")

    # Load HW and no_HW data
    print("Loading HW and no_HW data...")
    df_hw = pd.read_feather(os.path.join(summary_dir, f'HW{hw_threshold}.feather'))
    df_no_hw = pd.read_feather(os.path.join(summary_dir, f'no_hw_HW{hw_threshold}.feather'))
    print("HW and no_HW data loaded.")

    # Calculate average for no_HW data
    print("Calculating average for no_HW data...")
    df_no_hw_avg = df_no_hw[['lat', 'lon', 'year', 'hour', 'UHI', 'UWBI']].groupby(['lat', 'lon', 'year', 'hour']).mean().reset_index()
    print("Average for no_HW data calculated.")

    # Calculate UHI difference
    local_hour_adjusted_df = calculate_uhi_diff(df_hw, df_no_hw_avg)

    # Load location ID and height data
    print("Loading location ID and height data...")
    location_ID_path = os.path.join(summary_dir, 'location_IDs.nc')
    heightdat = os.path.join(summary_dir, 'topodata_0.9x1.25_USGS_070110_stream_c151201.nc')

    ds_location_ID = xr.open_dataset(location_ID_path, engine='netcdf4')
    ds_height = xr.open_dataset(heightdat, engine='netcdf4')
    print("Location ID and height data loaded.")

    # Merge TOPO into location_ID dataset
    print("Merging TOPO data...")
    ds_merged = xr.merge([ds_location_ID, ds_height.TOPO.isel(time=0)]).drop('time')

    # Convert to DataFrame
    merged_df = ds_merged[['location_ID', 'TOPO']].to_dataframe().reset_index()
    print("TOPO data merged and converted to DataFrame.")

    # Merge TOPO values with existing DataFrame
    print("Merging TOPO values with existing DataFrame...")
    local_hour_adjusted_df = local_hour_adjusted_df.merge(merged_df[['location_ID', 'TOPO']], 
                                                          on='location_ID', how='left', validate='m:1')
    print("TOPO values merged with existing DataFrame.")

    # Save the updated DataFrame
    output_path = os.path.join(summary_dir, f'local_hour_adjusted_variables_HW{hw_threshold}.feather')
    local_hour_adjusted_df.to_feather(output_path)
    print(f"Saved updated DataFrame to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HW and no_HW data and add TOPO information.")
    parser.add_argument("--summary_dir", type=str, 
                        default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/hw95_summary",
                        help="Directory containing the summary data files.")
    parser.add_argument("--hw_threshold", type=int, choices=[90, 95], required=True,
                        help="Heatwave threshold (90 or 95)")
    args = parser.parse_args()

    main(args.summary_dir, args.hw_threshold)