import xarray as xr
import pandas as pd
import numpy as np
import os
import duckdb

import cftime

# Set paths for input and output files
netcdf_file = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/hw95_summary/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h1.hwdaysOnly.nc'
output_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/hw95_summary'
summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'
output_file = os.path.join(output_dir, 'hw_data.feather')

def load_and_process_netcdf(file_path, variables):
    """
    Load NetCDF file, extract specified variables, and process the data.

    Args:
        file_path (str): Path to the NetCDF file.
        variables (list): List of variable names to extract from the NetCDF file.

    Returns:
        pd.DataFrame: Processed DataFrame containing the specified variables.
    """
    # Open the NetCDF file
    ds = xr.open_dataset(file_path)
    
    # Convert to DataFrame and reset index
    df = ds[variables].to_dataframe().reset_index()
    
    # Remove rows with missing data for key variables
    df = df.dropna(subset=['TSA_U', 'TREFMXAV_R'])
    
    # Convert cftime to pandas datetime
    def convert_cftime_to_datetime(ct):
        return pd.Timestamp(ct.year, ct.month, ct.day)
    
    df['time'] = df['time'].apply(convert_cftime_to_datetime)
    
    #todo: I need to create this HW flag in boolean
    df['HW'] = df['HW'].notna() & (df['HW'] != 0)
    
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
    Calculate heatwave days based on a given percentile threshold.

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
    rolling_sum = exceed_threshold.groupby(df['location_ID']).rolling(window=3, min_periods=3).sum().reset_index(level=0, drop=True)
    
    # Create heatwave column name based on percentile
    hw_column = f'HW{int(percentile*100)}'
    
    # Set heatwave days where the rolling sum is >= 3 and the threshold is exceeded
    df[hw_column] = (rolling_sum >= 3) & exceed_threshold
    
    # Fill NaN values with False (for the first two days of each location that can't be part of a 3-day streak)
    df[hw_column] = df[hw_column].fillna(False)
    
    # Add the threshold column
    df[f'threshold{int(percentile*100)}'] = threshold
    
    return df

# Main execution
if __name__ == "__main__":
    # Define variables to extract from NetCDF
    variables = ['TSA', 'TSA_U', 'TSA_R', 'TREFMXAV_R', 'HW']
    
    # Load and process NetCDF data
    print("Loading and processing NetCDF data...")
    df = load_and_process_netcdf(netcdf_file, variables)
    
    # Load location data
    print("Loading location data...")
    location_df = load_location_data(os.path.join(summary_dir, 'location_IDs.nc'))
    
    # Merge location data with main DataFrame
    print("Merging location data...")
    df = pd.merge(df, location_df, on=['lat', 'lon'], how='left')
    df = df.sort_values(['location_ID', 'time'])
    
    # Calculate heatwave days for 95th and 90th percentiles
    print("Calculating heatwave 95 threshold days...")
    df = calculate_heatwave_days(df, 0.95)
    print("Calculating heatwave 95 threshold days...")
    df = calculate_heatwave_days(df, 0.90)
    
    # Save processed data to feather file
    print(f"Saving processed data to {output_file}...")
    df.to_feather(output_file)
    print("Data saved successfully.")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 20)
    print("DataFrame Info:")
    print(df.info())
    print(f"\nNumber of unique lat-lon pairs: {df[['lat', 'lon']].drop_duplicates().shape[0]}")
    print(f"Number of unique lat-lon pairs with HW = 1: {df[df['HW'] == 1][['lat', 'lon']].drop_duplicates().shape[0]}")
    
    # Calculate HW ratios using duckdb for efficiency
    print("\nHeatwave Ratios:")
    print("-" * 20)
    hw_ratio = duckdb.query("SELECT SUM(CAST(HW AS FLOAT)) / COUNT(*) AS HW_ratio FROM df").to_df()
    hw95_ratio = df['HW95'].mean()
    hw90_ratio = df['HW90'].mean()
    
    print(f"HW ratio:  {hw_ratio['HW_ratio'][0]:.4f}")
    print(f"HW95 ratio: {hw95_ratio:.4f}")
    print(f"HW90 ratio: {hw90_ratio:.4f}")

    print("\nProcessing complete.")