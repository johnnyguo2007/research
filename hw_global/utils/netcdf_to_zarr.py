import xarray as xr
import os
import pandas as pd
import zarr
import dask


# Setup paths and variables
netcdf_dir = '/tmpdata/i.e215.I2000Clm50SpGs.hw_production.02/'
zarr_path = '/tmpdata/zarr/i.e215.I2000Clm50SpGs.hw_production.02'
frequent_vars = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8']  # Your frequently accessed variables
other_vars = ['var9', 'var10', ...]  # The rest of your variables

# Function to preprocess and split the dataset
def preprocess_and_split_dataset(filename, frequent_vars, other_vars):
    ds = xr.open_dataset(filename, chunks={'time': 24})  # Assuming daily files with hourly data
    freq_ds = ds[frequent_vars]
    other_ds = ds[other_vars]
    return freq_ds, other_ds

# Function to append data to Zarr with specified chunking and compression
def append_to_zarr(ds, zarr_group, chunk_shape):
    encoding = {var: {'chunks': chunk_shape, 'compressor': zarr.Blosc(cname='zstd', clevel=3)} for var in ds.data_vars}
    ds.to_zarr(zarr_group, mode='a', append_dim='time', encoding=encoding, consolidated=True)

# Function to process and convert NetCDF files to Zarr
def process_files(netcdf_filenames, zarr_store, frequent_vars, other_vars, freq_chunk_shape, other_chunk_shape):
    for i, filename in enumerate(netcdf_filenames):
        print(f'Processing file {i+1}/{len(netcdf_filenames)}: {filename}')
        freq_ds, other_ds = preprocess_and_split_dataset(filename, frequent_vars, other_vars)

        # Determine the mode for writing to Zarr ('w' for first file, 'a' for appending)
        mode = 'w' if i == 0 else 'a'

        # Write the frequent and other variables to their respective Zarr groups
        if not freq_ds.variables:
            print("No frequent variables found in this dataset.")
        else:
            append_to_zarr(freq_ds, os.path.join(zarr_store, 'frequent_vars'), freq_chunk_shape)

        if not other_ds.variables:
            print("No other variables found in this dataset.")
        else:
            append_to_zarr(other_ds, os.path.join(zarr_store, 'other_vars'), other_chunk_shape)

# Generate a list of NetCDF filenames
start_date = pd.to_datetime('start_of_your_dataset')
end_date = pd.to_datetime('end_of_your_dataset')
date_range = pd.date_range(start_date, end_date, freq='D')
netcdf_filenames = [os.path.join(netcdf_dir, f'data_{date.strftime("%Y-%m-%d")}.nc') for date in date_range]

# Chunk shapes: (time, variables) for frequent, (time, 1) for others, assuming 24 hours per file
freq_chunk_shape = (24, len(frequent_vars))
other_chunk_shape = (24, 1)

# Process the files
process_files(netcdf_filenames, zarr_path, frequent_vars, other_vars, freq_chunk_shape, other_chunk_shape)
