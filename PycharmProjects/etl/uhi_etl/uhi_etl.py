import os
import glob
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import yaml

import psutil
import time
from datetime import datetime

################ Constants ################
# Define global constants
LAT_LEN: int = 192
LON_LEN: int = 288
fahrenheit_threshold: float = 90
kelvin_threshold: float = 305.372
###########################################

def time_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_dt = datetime.now()  # Capture the datetime at start

        result = func(*args, **kwargs)

        end_time = time.time()
        end_dt = datetime.now()  # Capture the datetime at end

        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format the output string
        time_str = f"{int(hours):02d}h:{int(minutes):02d}m:{seconds:.2f}s"
        print(
            f"{func.__name__} started at {start_dt.strftime('%Y-%m-%d %H:%M:%S')} and finished at {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{func.__name__} executed in {time_str}")

        return result

    return wrapper

def looks_like_file(path):
    # A simple heuristic: if the last segment after a split on the OS separator has a dot, it might be a file.
    return '.' in os.path.basename(path)


def ensure_directory_exists(file_path):
    # Extract the directory part of the file path
    if looks_like_file(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path

    # Check if the directory exists
    if not os.path.exists(directory):
        # If the directory does not exist, create it
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    # else:
    #     print(f"Directory '{directory}' already exists.")


def substitute_variables(data, parent_data=None):
    if parent_data is None:
        parent_data = data  # Initial call uses the root of the data

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.format(**parent_data)
            else:
                substitute_variables(value, parent_data)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                data[i] = item.format(**parent_data)
            else:
                substitute_variables(item, parent_data)


def show_memory_usage(message):
    # Get the current process
    process = psutil.Process(os.getpid())

    # Get the memory usage in bytes
    memory_info = process.memory_info()

    # Convert bytes to megabytes
    memory_usage_mb = memory_info.rss / (1024 * 1024)

    # Print the current memory usage
    print(f"{message} memory usage: {memory_usage_mb:.2f} MB")


def load_config(config_file):
    """Loads configuration parameters from a YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Perform variable substitution
    substitute_variables(config)
    return config


def extract_variables(input_pattern, variables, output_file, log_file_path):
    """Extracts specified variables from input files and saves them to an output file."""
    print(f"Extracting variables {variables} from {input_pattern} into {output_file}")
    file_paths = sorted(glob.glob(input_pattern))

    for file_path in file_paths:
        log_file_status(log_file_path, file_path, "Processed")

    #todo: should add in drop_variables to avoid the cost of joining and potentially reduce memory usage
    ds = xr.open_mfdataset(file_paths, combine='by_coords')
    ds_var = ds[variables]
    ensure_directory_exists(output_file)
    ds_var.to_netcdf(output_file)


def convert_time(ds):
    """Converts time values in the dataset to pandas Timestamps."""
    ds['time'] = [pd.Timestamp(time.strftime()) for time in ds['time'].values]
    return ds


def round_to_nearest_hour(time_values):
    """Rounds cftime.DatetimeNoLeap objects to the nearest hour."""
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
    """Sets unwanted data to NaN while keeping the dataset structure."""
    condition_jja_nh = (ds['time.season'] == 'JJA') & (ds['lat'] >= 0)
    condition_djf_sh = (ds['time.season'] == 'DJF') & (ds['lat'] < 0)
    condition_tsa_u_not_null = ds['TSA_U'].notnull()
    condition = (condition_jja_nh | condition_djf_sh) & condition_tsa_u_not_null
    ds_filtered = ds.where(condition)
    return ds_filtered


def log_file_status(log_file_path, file_path, status):
    ensure_directory_exists(log_file_path)
    """Logs the status of each file."""
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_path} - {status}\n')


def append_to_zarr(ds, zarr_group):
    """Appends data to a Zarr group."""
    chunk_size = {'time': 24 * 3 * 31, 'lat': 96, 'lon': 144}
    ds = ds.chunk(chunk_size)
    if os.path.exists(zarr_group):
        ds.to_zarr(zarr_group, mode='a', append_dim='time', consolidated=True)
    else:
        encoding = {var: {'compressor': zarr.Blosc(cname='zstd', clevel=3)} for var in ds.data_vars}
        ds.to_zarr(zarr_group, mode='w', encoding=encoding, consolidated=True)


def netcdf_to_zarr_process_year(netcdf_dir, zarr_path, log_file_path, year, var_list, ds_daily_grid_hw):
    """Processes data for a specific year and appends it to the Zarr group."""
    file_pattern = os.path.join(netcdf_dir, f'i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.{year}-*-00000.nc')
    file_paths = sorted(glob.glob(file_pattern))

    print("Processing", file_pattern)

    if not file_paths:
        log_file_status(log_file_path, f'No files found for {file_pattern}', "Missing")
        return

    ds_year = xr.open_mfdataset(file_paths, chunks={'time': 24 * 31})[var_list]
    ds_daily_grid_hw_year = ds_daily_grid_hw.sel(time=ds_daily_grid_hw['time'].dt.year == year).compute()

    ds_year['time'] = round_to_nearest_hour(ds_year['time'].values)
    ds_year = convert_time(ds_year)
    ds_year = set_unwanted_to_nan(ds_year)

    overlapping_mask_year = (
            ds_daily_grid_hw_year['HW'].sel(time=slice(ds_year['time'].min(), ds_year['time'].max())) == 1)
    full_hourly_range_year = pd.date_range(start=ds_year['time'].min().values, end=ds_year['time'].max().values,
                                           freq='H')
    # Expand the daily mask to hourly using forward fill for the current year
    hourly_mask_year = overlapping_mask_year.reindex(time=full_hourly_range_year, method='ffill').compute()

    # Apply the hourly mask to ds_year to get filtered data for the current year
    ds_year['HW'] = hourly_mask_year

    append_to_zarr(ds_year, os.path.join(zarr_path, '3Dvars'))


def separate_hw_no_hw_process_in_chunks(ds, chunk_size, zarr_path):
    """Processes data in smaller chunks and separates HW and No HW data."""
    num_time_steps = ds.dims['time']

    for start in range(0, num_time_steps, chunk_size):
        end = start + chunk_size
        print(f"Processing time steps {start} to {min(end, num_time_steps)}")

        ds_chunk = ds.isel(time=slice(start, end))

        hw_computed = ds_chunk.HW.compute()

        ds_hw_chunk = ds_chunk.where(hw_computed).compute()
        ds_no_hw_chunk = ds_chunk.where(~hw_computed).compute()

        print(f"Appending HW to Zarr", ds_hw_chunk.time.values[0], ds_hw_chunk.time.values[-1])
        append_to_zarr(ds_hw_chunk, os.path.join(zarr_path, 'HW'))
        print(f"Appending No HW to Zarr", ds_no_hw_chunk.time.values[0], ds_no_hw_chunk.time.values[-1])
        append_to_zarr(ds_no_hw_chunk, os.path.join(zarr_path, 'NO_HW'))


def zarr_to_dataframe(zarr_path, start_year, end_year, output_path, hw_flag, core_vars=None):
    """Converts Zarr data to DataFrame format, processing monthly but saves it as annual Parquet files."""
    ds = xr.open_zarr(os.path.join(zarr_path, hw_flag), chunks='auto')
    if core_vars:
        ds = ds[core_vars]

    ensure_directory_exists(output_path)
    for year in range(start_year, end_year + 1):
        df_list = []
        for month in range(1, 13):  # Loop through each month
            month_str = f'{year}-{month:02d}'  # Format as 'YYYY-MM'
            ds_month = ds.sel(time=month_str)  # Select all days for given month and year
            df_month = ds_month.to_dataframe().dropna()  # Convert to DataFrame and drop NA values
            df_list.append(df_month)

        df_year = pd.concat(df_list)
        df_year.columns = df_year.columns.str.replace('UBWI', 'UWBI')

        parquet_file_path = os.path.join(output_path, f'ALL_{hw_flag}_{year}.parquet')
        df_year.to_parquet(parquet_file_path, engine='pyarrow', index=True)

        print(f'Saved {year} data to {parquet_file_path}')
        show_memory_usage(f"Memory usage after saving {year} data")


@time_func
def load_data_from_parquet(data_dir, start_year, end_year, col_list=None):
    """Load HW and NO_HW data from the specified directory for a range of years."""
    df_hw_list = []
    df_no_hw_list = []

    for year in range(start_year, end_year + 1):
        file_name_hw = os.path.join(data_dir, f"ALL_HW_{year}.parquet")
        file_name_no_hw = os.path.join(data_dir, f"ALL_NO_HW_{year}.parquet")
        if not col_list:
            df_hw_list.append(pd.read_parquet(file_name_hw))
            df_no_hw_list.append(pd.read_parquet(file_name_no_hw))
        else:
            df_hw_list.append(pd.read_parquet(file_name_hw, columns=col_list))
            df_no_hw_list.append(pd.read_parquet(file_name_no_hw, columns=col_list))

    return pd.concat(df_hw_list), pd.concat(df_no_hw_list)


def add_year_month_hour_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure 'time' is of datetime type
    # df.index = df.index.set_levels(
    #     [df.index.levels[0], df.index.levels[1], pd.to_datetime(df.index.levels[2])])
    """Decompose the 'time' index of DataFrame into 'hour', 'month', and 'year' and append them as columns."""
    df['hour'] = df.index.get_level_values('time').hour
    df['month'] = df.index.get_level_values('time').month
    df['year'] = df.index.get_level_values('time').year
    return df


def check_data_overlap(df_hw, df_no_hw):
    """Check if there is any overlap in MultiIndexes between two DataFrames."""
    keys_hw = set(df_hw.index)
    keys_no_hw = set(df_no_hw.index)

    return keys_hw & keys_no_hw


@time_func
def calculate_uhi_diff(df_hw, df_no_hw_avg):
    """Calculate the difference between UHI values of HW and average NO_HW on matching columns."""
    df_hw_reset = df_hw.reset_index()
    df_no_hw_avg_reset = df_no_hw_avg.reset_index()
    merged_df = pd.merge(df_hw_reset, df_no_hw_avg_reset[['lat', 'lon', 'year', 'hour', 'UHI', 'UWBI']],
                         on=['lat', 'lon', 'year', 'hour'],
                         suffixes=('', '_avg'))
    merged_df['UHI_diff'] = merged_df['UHI'] - merged_df['UHI_avg']
    merged_df['UWBI_diff'] = merged_df['UWBI'] - merged_df['UWBI_avg']
    return merged_df


@time_func
def convert_time_to_local_and_add_hour(df):
    """Adjusts DataFrame time data to local based on longitude and extracts local hour."""

    def calculate_timezone_offset(longitude):
        return np.floor(longitude / 15.0).astype(int)

    df = df.reset_index()
    offsets = calculate_timezone_offset(df['lon'].values)
    df['local_time'] = df['time'] + pd.to_timedelta(offsets, unit='h')
    df['local_hour'] = df['local_time'].dt.hour
    return df

@time_func
def add_event_id(df):
    """Add event_ID to the DataFrame."""
    df.sort_values(by=['location_ID', 'time'], inplace=True)
    df['time_diff'] = df.groupby('location_ID')['time'].diff().dt.total_seconds() / 3600
    df['new_event'] = (df['time_diff'] > 1)
    df['event_ID'] = df.groupby('location_ID')['new_event'].cumsum()
    df['global_event_ID'] = df['location_ID'].astype(str) + '_' + df['event_ID'].astype(str)
    return df
