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

    return np.array(rounded_times)


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


def process_year(netcdf_dir, zarr_path, log_file_path, year):
    file_pattern = os.path.join(netcdf_dir,
                                f'i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.{year}-*-00000.nc')
    file_paths = sorted(glob.glob(file_pattern))

    print("processing " , file_pattern)

    if not file_paths:
        log_file_status(log_file_path, f'No files found for {file_pattern}', "Missing")
        return

    ds = xr.open_mfdataset(file_paths, chunks={'time': 24 * 31})
    ds['time'] = round_to_nearest_hour(ds['time'].values)
    append_to_zarr(ds, os.path.join(zarr_path, '3Dvars'))


# def process_year(netcdf_dir, zarr_path, log_file_path, year):
#     for month in range(1, 13):
#         process_month(netcdf_dir, zarr_path, log_file_path, year, month)


if __name__ == "__main__":
    netcdf_dir = '/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/hourly_raw'
    zarr_path = '/home/jguo/process_data/zarr/i.e215.I2000Clm50SpGs.hw_production.02'
    output_dir = zarr_path
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'processed_files.log')

    start_year = 1985
    end_year = 1986

    for year in range(start_year, end_year + 1):
        process_year(netcdf_dir, zarr_path, log_file_path, year)
