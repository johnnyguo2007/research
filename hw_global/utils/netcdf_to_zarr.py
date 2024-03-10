import xarray as xr
import os
import pandas as pd
import numpy as np
import zarr

def generate_noleap_date_range(start_date, end_date):
    """Generate a date range for a 'noleap' calendar."""
    current_date = start_date
    date_list = []
    while current_date <= end_date:
        date_list.append(current_date)
        next_day = current_date + pd.Timedelta(days=1)
        if next_day.month == 2 and next_day.day == 29:
            next_day += pd.Timedelta(days=1)
        current_date = next_day
    return date_list

def initialize_zarr_group(zarr_store, group_name, chunk_shape, vars, dims):
    """Initialize Zarr groups before processing to avoid append_dim issues."""
    group_path = os.path.join(zarr_store, group_name)
    if not os.path.exists(group_path):
        os.makedirs(group_path, exist_ok=True)
        ds = xr.Dataset({
            var: (dims, np.zeros((1, *chunk_shape[1:]), dtype=float)) for var in vars
        })
        encoding = {var: {'chunks': chunk_shape, 'compressor': zarr.Blosc(cname='zstd', clevel=3)} for var in vars}
        ds.to_zarr(group_path, mode='w', encoding=encoding, consolidated=True)

def preprocess_and_split_dataset(filename, frequent_vars, other_vars):
    try:
        ds = xr.open_dataset(filename)
        freq_ds = ds[frequent_vars]
        other_ds = ds.drop_vars(frequent_vars)
        return freq_ds, other_ds
    except Exception as e:
        print(f"Error preprocessing file {filename}: {e}")
        return None, None

def append_to_zarr(ds, zarr_group, chunk_shape):
    if ds is not None and ds.variables:
        encoding = {var: {'chunks': chunk_shape, 'compressor': zarr.Blosc(cname='zstd', clevel=3)} for var in ds.data_vars}
        ds.to_zarr(zarr_group, mode='a', append_dim='time', encoding=encoding, consolidated=True)

def process_single_file(filename, zarr_store, frequent_vars, other_vars, chunk_shape):
    print(f'Processing file: {filename}')
    freq_ds, other_ds = preprocess_and_split_dataset(filename, frequent_vars, other_vars)
    append_to_zarr(freq_ds, os.path.join(zarr_store, 'frequent_vars'), chunk_shape)
    if other_ds:
        append_to_zarr(other_ds, os.path.join(zarr_store, 'other_vars'), chunk_shape)

def process_files(netcdf_filenames, zarr_store, frequent_vars, other_vars, chunk_shape):
    for filename in netcdf_filenames:
        process_single_file(filename, zarr_store, frequent_vars, other_vars, chunk_shape)

if __name__ == "__main__":
    netcdf_dir = '/tmpdata/i.e215.I2000Clm50SpGs.hw_production.02/'
    zarr_path = '/tmpdata/zarr/i.e215.I2000Clm50SpGs.hw_production.02'

    start_date = pd.to_datetime('1986-01-01')
    end_date = pd.to_datetime('1986-12-31')
    date_range = generate_noleap_date_range(start_date, end_date)
    netcdf_filenames = [os.path.join(netcdf_dir, f"i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.{date.strftime('%Y-%m-%d')}-00000.nc") for date in date_range]

    core_vars = ['TSA', 'TSA_R', 'TSA_U', 'WBA', 'WBA_R', 'WBA_U']
    # Example to fetch other variables, adjust as per your data
    one_hourly_file = '/tmpdata/i.e215.I2000Clm50SpGs.hw_production.02/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.1986-01-01-00000.nc'
    ds_hourly = xr.open_dataset(one_hourly_file)
    other_vars = [var for var in ds_hourly.data_vars if (var not in core_vars and len(var.dims) == 3)]


    chunk_shape_3D = (8760, 192, 288)  # (time, lat, lon)
    dims_3D = ['time', 'lat', 'lon']

    # Initialize Zarr groups for core and other variables
    initialize_zarr_group(zarr_path, 'frequent_vars', chunk_shape_3D, core_vars, dims_3D)
    initialize_zarr_group(zarr_path, 'other_vars', chunk_shape_3D, other_vars, dims_3D)

    process_files(netcdf_filenames, zarr_path, core_vars, other_vars, chunk_shape_3D)