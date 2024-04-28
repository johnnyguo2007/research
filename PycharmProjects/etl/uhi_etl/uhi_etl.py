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
from utils import *




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



def set_unwanted_to_nan(ds):
    """Sets unwanted data to NaN while keeping the dataset structure."""
    condition_jja_nh = (ds['time.season'] == 'JJA') & (ds['lat'] >= 0)
    condition_djf_sh = (ds['time.season'] == 'DJF') & (ds['lat'] < 0)
    condition_tsa_u_not_null = ds['TSA_U'].notnull()
    condition = (condition_jja_nh | condition_djf_sh) & condition_tsa_u_not_null
    ds_filtered = ds.where(condition)
    return ds_filtered


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
