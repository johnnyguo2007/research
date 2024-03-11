import os
import glob
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import zarr


def round_to_nearest_hour(time_values):
    # Function to round cftime.DatetimeNoLeap objects to the nearest hour
    rounded_times = []
    for time_value in time_values:
        # Extract year, month, day, and hour
        year, month, day, hour = time_value.year, time_value.month, time_value.day, time_value.hour
        # Extract minute to decide whether to round up or down
        minute = time_value.minute

        # If minute >= 30, round up to the next hour
        if minute >= 30 and hour < 23:
            hour += 1
        elif minute >= 30 and hour == 23:
            # Special case for end of the day, create a new datetime for the next day
            new_day = cftime.DatetimeNoLeap(year, month, day) + cftime.timedelta(days=1)
            year, month, day = new_day.year, new_day.month, new_day.day
            hour = 0

        # Construct new cftime.DatetimeNoLeap object with rounded hour
        rounded_time = cftime.DatetimeNoLeap(year, month, day, hour)
        rounded_times.append(rounded_time)

    #return pd.Series([pd.Timestamp(rt.year, rt.month, rt.day, rt.hour) for rt in rounded_times])
    return np.array(rounded_times)

# Function to set unwanted data to NaN while keeping the dataset structure
def set_unwanted_to_nan(ds):
    # Condition for JJA in the Northern Hemisphere
    condition_jja_nh = (ds['time.season'] == 'JJA') & (ds['lat'] >= 0)

    # Condition for DJF in the Southern Hemisphere
    condition_djf_sh = (ds['time.season'] == 'DJF') & (ds['lat'] < 0)

    # Set grid cells to NaN where TSA_U is null
    condition_tsa_u_not_null = ds['TSA_U'].notnull()

    # Combine conditions for the desired data, set others to NaN
    condition = (condition_jja_nh | condition_djf_sh) & condition_tsa_u_not_null

    # Apply condition, keeping structure intact
    ds_filtered = ds.where(condition)

    return ds_filtered


def log_file_status(log_file_path, file_path, status):
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_path} - {status}\n')


def append_to_zarr(ds, zarr_group):
    chunk_size = {'time': 24 * 3 * 31, 'lat': 96, 'lon': 144}
    ds = ds.chunk(chunk_size)
    if os.path.exists(zarr_group):
        ds.to_zarr(zarr_group, mode='a', append_dim='time', consolidated=True)
    else:
        encoding = {var: {'compressor': zarr.Blosc(cname='zstd', clevel=3)} for var in ds.data_vars}
        ds.to_zarr(zarr_group, mode='w', encoding=encoding, consolidated=True)


def process_year(netcdf_dir, zarr_path, log_file_path, year, var_list):
    file_pattern = os.path.join(netcdf_dir, f'i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.{year}-*-00000.nc')
    file_paths = sorted(glob.glob(file_pattern))

    print("Processing", file_pattern)

    if not file_paths:
        log_file_status(log_file_path, f'No files found for {file_pattern}', "Missing")
        return

    ds = xr.open_mfdataset(file_paths, chunks={'time': 24 * 31})[var_list]

    # Store data types before modification
    data_types_before = {var: ds[var].dtype for var in ds.data_vars}
    print("Data types before set_unwanted_to_nan:")
    for var, dtype in data_types_before.items():
        print(f'{var}: {dtype}')

    ds['time'] = round_to_nearest_hour(ds['time'].values)
    ds = set_unwanted_to_nan(ds)

    # Store data types after modification
    data_types_after = {var: ds[var].dtype for var in ds.data_vars}
    print("Data types after set_unwanted_to_nan:")
    for var, dtype in data_types_after.items():
        print(f'{var}: {dtype}')

    # Compare data types before and after
    print("\nData type changes:")
    for var in ds.data_vars:
        if data_types_before[var] != data_types_after[var]:
            print(f'{var}: {data_types_before[var]} -> {data_types_after[var]}')

    append_to_zarr(ds, os.path.join(zarr_path, '3Dvars'))




# def process_year(netcdf_dir, zarr_path, log_file_path, year):
#     for month in range(1, 13):
#         process_month(netcdf_dir, zarr_path, log_file_path, year, month)


if __name__ == "__main__":
    netcdf_dir = '/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/hourly_raw'
    zarr_path = '/home/jguo/process_data/zarr/test02'
    output_dir = zarr_path
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'processed_files.log')

    one_hourly_file = '/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/hourly_raw/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.1985-07-01-00000.nc'
    ds_hourly = xr.open_dataset(one_hourly_file)
    vars = [var for var in ds_hourly.data_vars if (len(ds_hourly[var].dims) == 3)]

    start_year = 1988
    end_year = 1999

    for year in range(start_year, end_year + 1):
        process_year(netcdf_dir, zarr_path, log_file_path, year, vars)
