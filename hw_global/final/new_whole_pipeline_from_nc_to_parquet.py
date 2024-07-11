import os
import glob
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
# import cartopy.crs as ccrs


def convert_time(ds):
    """
    Converts time values in the dataset to pandas Timestamps.
    """
    ds['time'] = [pd.Timestamp(time.strftime()) for time in ds['time'].values]
    return ds


def round_to_nearest_hour(time_values):
    """
    Rounds cftime.DatetimeNoLeap objects to the nearest hour.
    """
    rounded_times = []
    for time_value in time_values:
        year, month, day, hour = time_value.year, time_value.month, time_value.day, time_value.hour
        minute = time_value.minute

        if minute >= 30 and hour < 23:
            hour += 1
        elif minute >= 30 and hour == 23:
            new_day = cftime.DatetimeNoLeap(year, month, day) + cftime.timedelta(days=1)
            year, month, day = new_day.year, new_day.month, new_day.day
            hour = 0

        rounded_time = cftime.DatetimeNoLeap(year, month, day, hour)
        rounded_times.append(rounded_time)

    return np.array(rounded_times)


def set_unwanted_to_nan(ds):
    """
    Sets unwanted data to NaN while keeping the dataset structure.
    """
    condition_jja_nh = (ds['time.season'] == 'JJA') & (ds['lat'] >= 0)
    condition_djf_sh = (ds['time.season'] == 'DJF') & (ds['lat'] < 0)
    condition_tsa_u_not_null = ds['TSA_U'].notnull()
    condition = (condition_jja_nh | condition_djf_sh) & condition_tsa_u_not_null
    ds_filtered = ds.where(condition)
    return ds_filtered


def log_file_status(log_file_path, file_path, status):
    """
    Logs the status of each file.
    """
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_path} - {status}\n')


def process_data_for_month(sim_results_dir, log_file_path, year, month, var_list):
    """
    Processes data for a specific month, one file at a time, and returns a list of DataFrames.
    """
    hourly_pattern = f"i.e215.I2000Clm50SpGs.hw_production.05.clm2.h2.{year}-{month:02d}-*-00000.nc"
    file_pattern = os.path.join(sim_results_dir, hourly_pattern)
    file_paths = sorted(glob.glob(file_pattern))

    print(f"Processing {year}-{month:02d}")

    if not file_paths:
        log_file_status(log_file_path, f'No files found for {file_pattern}', "Missing")
        return None

    df_list = []

    for file_path in file_paths:
        # Process one file at a time
        print(f"Processing {file_path}")
        ds = xr.open_dataset(file_path)[var_list]

        ds['time'] = round_to_nearest_hour(ds['time'].values)
        ds = convert_time(ds)
        ds = set_unwanted_to_nan(ds)

        # Convert to DataFrame without resetting the index
        df = ds.to_dataframe().dropna()

        # Append to list
        df_list.append(df)

        # Close the dataset to free up memory
        ds.close()

    return df_list



def netcdf_to_parquet(sim_results_dir, parquet_dir, log_file_path, start_year, end_year, var_list):
    """
    Converts NetCDF data to monthly Parquet format.
    """
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            df_list = process_data_for_month(sim_results_dir, log_file_path, year, month, var_list)
            
            if df_list is not None and len(df_list) > 0:
                # Concatenate all DataFrames for the month
                df_month = pd.concat(df_list)

                # Save monthly data
                parquet_file_path = os.path.join(parquet_dir, f'{year}_{month:02d}.parquet')
                df_month.to_parquet(parquet_file_path, engine='pyarrow')
                print(f'Saved data for {year}-{month:02d} to {parquet_file_path}')

def main():
    """
    Main function that orchestrates the entire process.
    """
    case_name = 'i.e215.I2000Clm50SpGs.hw_production.05'
    case_results_dir = '/Trex/case_results'
    # sim_results_dir = os.path.join(case_results_dir, case_name, 'sim_results')
    sim_results_dir = '/media/jguo/external_data/simulation_output/archive/case/i.e215.I2000Clm50SpGs.hw_production.05/lnd/hist'

    research_results_summary_dir = os.path.join(case_results_dir, case_name, 'research_results/summary')
    os.makedirs(research_results_summary_dir, exist_ok=True)

    research_results_parquet_dir = os.path.join(case_results_dir, case_name, 'research_results/parquet')
    os.makedirs(research_results_parquet_dir, exist_ok=True)

    log_file_path = os.path.join(research_results_summary_dir, 'processed_files.log')
    start_year = 1985
    end_year = 2013

    monthly_pattern = r"i\.e215\.I2000Clm50SpGs\.hw_production\.05\.clm2\.h0\.(?P<YYYY>\d{4})-(?P<mm>\d{2})\.nc"
    daily_pattern = r"i\.e215\.I2000Clm50SpGs\.hw_production\.05\.clm2\.h1\.(?P<YYYY>\d{4})-(?P<mm>\d{2})-(?P<dd>\d{2})-00000\.nc"
    hourly_pattern = r"i\.e215\.I2000Clm50SpGs\.hw_production\.05\.clm2\.h2\.(?P<YYYY>\d{4})-(?P<mm>\d{2})-(?P<dd>\d{2})-00000\.nc"

    # Get list of variables from one hourly file
    one_hourly_file = os.path.join(sim_results_dir, f'{case_name}.clm2.h2.1985-07-01-00000.nc')
    ds_one_hourly_data = xr.open_dataset(one_hourly_file)
    var_list = [var for var in ds_one_hourly_data.data_vars if (len(ds_one_hourly_data[var].dims) == 3)]

    # Convert NetCDF to Parquet
    netcdf_to_parquet(sim_results_dir, research_results_parquet_dir, log_file_path,
                      start_year, end_year, var_list)


if __name__ == "__main__":
    main()