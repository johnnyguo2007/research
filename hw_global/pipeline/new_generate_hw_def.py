import xarray as xr
import pandas as pd
import numpy as np
import os
import duckdb
import logging

import cftime

# Set paths for input and output files
output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'
summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'
output_file_95 = os.path.join(output_dir, 'hw_data_95.feather')
output_file_90 = os.path.join(output_dir, 'hw_data_90.feather')
daily_feather_file = os.path.join(summary_dir,
                                  'i.e215.I2000Clm50SpGs.hw_production.05.clm2.h1.TSA_UR_TREFMXAV_R.feather')


def load_and_process_feather(file_path, variables):
    """
    Load NetCDF file, extract specified variables, and process the data.

    Args:
        file_path (str): Path to the NetCDF file.
        variables (list): List of variable names to extract from the NetCDF file.

    Returns:
        pd.DataFrame: Processed DataFrame containing the specified variables.
    """
    # Open the feather file
    df = pd.read_feather(daily_feather_file)

    # Remove rows with missing data for key variables
    df = df.dropna(subset=['TSA_U', 'TREFMXAV_R'])

    return df


def load_location_data(file_path):
    """
    Load location data from NetCDF file.

    Args:
        file_path (str): Path to the location NetCDF file.

    Returns:
        pd.DataFrame: DataFrame containing location data.
    """
    location_ds = xr.open_dataset(file_path)
    return location_ds.to_dataframe().reset_index()


def calculate_heatwave_days(df, percentile):
    """
    df is NOT global summer filtered!
    Add heatwave columns based on a given percentile threshold.

    Args:
        df (pd.DataFrame): Input DataFrame containing temperature data.
        percentile (float): Percentile to use for threshold calculation (e.g., 0.95 for 95th percentile).

    Returns:
        pd.DataFrame: DataFrame with added heatwave indicator and threshold columns.
    """
    # Sort the dataframe by location_ID and time
    df = df.sort_values(['location_ID', 'time'])

    # Calculate the percentile threshold of TREFMXAV_R for each location_ID
    threshold = df.groupby('location_ID')['TREFMXAV_R'].transform(lambda x: x.quantile(percentile))

    # Create a boolean mask for days exceeding the threshold
    exceed_threshold = df['TREFMXAV_R'] > threshold

    # Use rolling window to identify streaks of 3 or more days
    rolling_sum = exceed_threshold.groupby(df['location_ID']).rolling(window=3, min_periods=3).sum().reset_index(
        level=0, drop=True)

    # Create heatwave column name based on percentile
    hw_column = f'HW{int(percentile * 100)}'

    # Set heatwave days where the rolling sum is >= 3 and the threshold is exceeded
    df[hw_column] = (rolling_sum >= 3) & exceed_threshold

    # Fill NaN values with False (for the first two days of each location that can't be part of a 3-day streak)
    df[hw_column] = df[hw_column].fillna(False)

    # Add the threshold column
    df[f'threshold{int(percentile * 100)}'] = threshold

    return df


def add_event_id(df, hw_column):
    """
    Add event IDs to HW data.

    Args:
        df (pd.DataFrame): Input DataFrame containing HW data.
        hw_column (str): Name of the HW column to use for event ID calculation.

    Returns:
        pd.DataFrame: DataFrame with added event ID and global event ID columns.
    """
    logging.info(f"Adding event IDs to HW data for {hw_column}")
    initial_count = len(df)
    df = df.sort_values(by=['location_ID', 'time'])
    df['time_diff'] = df.groupby('location_ID')['time'].diff().dt.total_seconds() / 3600
    df['new_event'] = (df['time_diff'] > 1) & (df[hw_column] == True)
    df['event_ID'] = df.groupby('location_ID')['new_event'].cumsum()
    df['global_event_ID'] = df['location_ID'].astype(str) + '_' + df['event_ID'].astype(str)
    logging.info(f"Added event IDs to {len(df)} rows")
    # Validation check
    if len(df) != initial_count:
        raise ValueError(f"Data loss detected during event ID addition. Expected {initial_count} rows, got {len(df)}")
    return df


# Main execution
if __name__ == "__main__":
    # Define variables to extract from NetCDF
    variables = ['TSA', 'TSA_U', 'TSA_R', 'TREFMXAV_R']

    # Load and process NetCDF data
    print("Loading and processing Daily Feather data...")
    df = load_and_process_feather(daily_feather_file, variables)

    # Load location data
    print("Loading location data...")
    location_df = load_location_data(os.path.join(summary_dir, 'location_IDs.nc'))

    # Merge location data with main DataFrame
    print("Merging location data...")
    df = pd.merge(df, location_df, on=['lat', 'lon'], how='left')
    df = df.sort_values(['location_ID', 'time'])

    # Calculate heatwave days for 95th and 90th percentiles
    print("Calculating heatwave 95 threshold days...")
    df_95 = calculate_heatwave_days(df, 0.95)
    print("Calculating heatwave 90 threshold days...")
    df_90 = calculate_heatwave_days(df, 0.90)

    # Add event IDs for 95th and 90th percentiles
    print("Adding event IDs for 95th percentile...")
    df_95 = add_event_id(df_95, 'HW95')
    print("Adding event IDs for 90th percentile...")
    df_90 = add_event_id(df_90, 'HW90')

    # Save processed data to feather files
    print(f"Saving processed data for 95th percentile to {output_file_95}...")
    df_95.to_feather(output_file_95)
    print(f"Saving processed data for 90th percentile to {output_file_90}...")
    df_90.to_feather(output_file_90)
    print("Data saved successfully.")

    print("\nProcessing complete.")