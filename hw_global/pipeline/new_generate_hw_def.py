import xarray as xr
import pandas as pd
import numpy as np
import os
import duckdb
import logging
import argparse

import cftime

# Set paths for input and output files
output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'
summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'
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
    Add heatwave columns based on a given percentile threshold.
    Args:
        df (pd.DataFrame): Input DataFrame containing temperature data.
        percentile (float): Percentile to use for threshold calculation (e.g., 0.95 for 95th percentile).
    Returns:
        pd.DataFrame: DataFrame with added heatwave indicator, threshold, and Nth_day columns.
    """
    # Sort the dataframe by location_ID and time
    df = df.sort_values(['location_ID', 'time'])

    # Calculate the percentile threshold of TREFMXAV_R for each location_ID
    thresholds = df.groupby('location_ID')['TREFMXAV_R'].transform(lambda x: x.quantile(percentile))

    # Create a new column 'exceeded' based on the threshold
    df['exceeded'] = df['TREFMXAV_R'] > thresholds

    # Function to detect heatwave days
    def detect_heatwave(group):
        # Create shifted columns for the two days before and after
        two_days_before = group.shift(2)
        one_day_before = group.shift(1)
        one_day_after = group.shift(-1)
        two_days_after = group.shift(-2)

        # Apply the heatwave conditions
        return group & (  # The day itself must have exceeded
                (two_days_before & one_day_before) |  # two days before are exceeded
                (one_day_after & two_days_after) |  # two days after are exceeded
                (one_day_before & one_day_after)  # one day before and one day after are exceeded
        )

    # Create heatwave column name based on percentile
    hw_column = f'HW{int(percentile * 100)}'

    # Apply the heatwave detection function and create the hw column
    df[hw_column] = df.groupby('location_ID')['exceeded'].transform(detect_heatwave)

    # Add the threshold column
    df[f'threshold{int(percentile * 100)}'] = thresholds

    # Calculate the event_id and Nth_day
    def assign_event_id(group):
        # Create a new group when HW changes from False to True
        event_changes = group != group.shift(1)
        # Cumulative sum of changes, but only for True values
        return (event_changes & group).cumsum()

    # Apply assign_event_id to each location group
    df['event_ID'] = df.groupby('location_ID')[hw_column].transform(assign_event_id)

    # Calculate Nth_day
    df['Nth_day'] = df.groupby(['location_ID', 'event_ID']).cumcount() + 1
    df.loc[~df[hw_column], 'Nth_day'] = 0

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

    # event_ID was created in calculate_heatwave_days
    df['global_event_ID'] = df['location_ID'].astype(str) + '_' + df['event_ID'].astype(str)
    logging.info(f"Added event IDs to {len(df)} rows")
    # Validation check
    if len(df) != initial_count:
        raise ValueError(f"Data loss detected during event ID addition. Expected {initial_count} rows, got {len(df)}")
    return df


# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process heatwave data for multiple thresholds.')
    parser.add_argument('--create', type=str, help='Comma-separated list of thresholds (e.g., 99,98,95,90)')
    args = parser.parse_args()

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

    # Process thresholds from command-line argument
    if args.create:
        thresholds = [float(t) / 100 for t in args.create.split(',')]
    else:
        thresholds = [0.95, 0.90]  # Default thresholds if not provided

    # Calculate heatwave days and add event IDs for each threshold
    for threshold in thresholds:
        print(f"Calculating heatwave {int(threshold * 100)} threshold days...")
        df_threshold = calculate_heatwave_days(df, threshold)

        print(f"Adding event IDs for {int(threshold * 100)}th percentile...")
        df_threshold = add_event_id(df_threshold, f'HW{int(threshold * 100)}')

        output_file = os.path.join(output_dir, f'hw_def_{int(threshold * 100)}.feather')
        print(f"Saving processed data for {int(threshold * 100)}th percentile to {output_file}...")
        df_threshold.to_feather(output_file)

    print("Data saved successfully.")
    print("\nProcessing complete.")