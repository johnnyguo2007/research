import os
import glob
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import zarr


def round_to_nearest_hour(time_values):
    rounded_times = []
    for time_value in time_values:
        if time_value.minute >= 30:
            time_value += cftime.timedelta(hours=1)
        rounded_times.append(time_value.replace(minute=0, second=0, microsecond=0))
    return np.array(rounded_times)


def log_file_status(log_file_path, file_path, status):
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_path} - {status}\n')


def append_to_zarr(ds, zarr_group):
    chunk_size = {'time': 720, 'lat': 96, 'lon': 144}
    ds = ds.chunk(chunk_size)
    if os.path.exists(zarr_group):
        ds.to_zarr(zarr_group, mode='a', append_dim='time', consolidated=True)
    else:
        encoding = {var: {'compressor': zarr.Blosc(cname='zstd', clevel=3)} for var in ds.data_vars}
        ds.to_zarr(zarr_group, mode='w', encoding=encoding, consolidated=True)


def process_month(netcdf_dir, zarr_path, log_file_path, year, month):
    file_pattern = os.path.join(netcdf_dir,
                                f'i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.{year}-{month:02d}-*-00000.nc')
    file_paths = sorted(glob.glob(file_pattern))

    if not file_paths:
        log_file_status(log_file_path, f'No files found for {year}-{month:02d}', "Missing")
        return

    ds = xr.open_mfdataset(file_paths, chunks={'time': 720})
    ds['time'] = round_to_nearest_hour(ds['time'].values)
    append_to_zarr(ds, os.path.join(zarr_path, '3Dvars'))


def process_year(netcdf_dir, zarr_path, log_file_path, year):
    for month in range(1, 13):
        process_month(netcdf_dir, zarr_path, log_file_path, year, month)


if __name__ == "__main__":
    output_dir = '/tmpdata/summerized_data/i.e215.I2000Clm50SpGs.hw_production.02/monthly_avg_for_each_year'
    os.makedirs(output_dir, exist_ok=True)

    netcdf_dir = '/tmpdata/i.e215.I2000Clm50SpGs.hw_production.02/'
    zarr_path = '/tmpdata/zarr/test2/'
    log_file_path = os.path.join(output_dir, 'processed_files.log')

    start_year = 1985
    end_year = 1986

    for year in range(start_year, end_year + 1):
        process_year(netcdf_dir, zarr_path, log_file_path, year)
