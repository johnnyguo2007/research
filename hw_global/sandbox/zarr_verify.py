#%%
import zarr
import os
import pandas as pd
import xarray as xr
import numpy as np
#%%
# Read the Zarr dataset using Xarray with automatic chunking for Dask
# ds = xr.open_zarr('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/zarr/3Dvars', chunks='auto')
ds = xr.open_zarr('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/zarr/3Dvars')
core_vars = ['TSA', 'TSA_R', 'TSA_U', 'WBA', 'WBA_R', 'WBA_U', 'HW']
ds = ds.sel(time=slice('1985-01-02', '1985-12-31'))


#%%
#read in netcdf file i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.1985-01-01-00000.nc
ds_netcdf = xr.open_dataset('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/sim_results/hourly/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.1985-01-01-00000.nc')

#%%
# Identify lat and lon values where the mask is True
mask = ds_netcdf.isel(time=0)['TSA_U'].notnull()
mask.sum()

#%%
ds['UHI'] = ds.TSA_U - ds.TSA_R
ds['UBWI'] = ds.WBA_U - ds.WBA_R

#%%

#%%

def append_to_zarr(ds, zarr_group):
    chunk_size = {'time': 24 * 3 *31, 'lat': 96, 'lon': 144}
    ds = ds.chunk(chunk_size)
    if os.path.exists(zarr_group):
        ds.to_zarr(zarr_group, mode='a', append_dim='time', consolidated=True)
    else:
        encoding = {var: {'compressor': zarr.Blosc(cname='zstd', clevel=3)} for var in ds.data_vars}
        ds.to_zarr(zarr_group, mode='w', encoding=encoding, consolidated=True)
#%%
import xarray as xr
import numpy as np

# Function to process data in smaller chunks
def process_in_chunks(ds, chunk_size, zarr_path):
    # Determine the number of time steps
    num_time_steps = ds.dims['time']

    # Iterate over the dataset in chunks
    for start in range(0, num_time_steps, chunk_size):
        end = start + chunk_size
        print(f"Processing time steps {start} to {min(end, num_time_steps)}")

        # Select the chunk
        ds_chunk = ds.isel(time=slice(start, end)) #.sel(lat=lat_values, lon=lon_values)

        # Compute the boolean indexer for the current chunk
        hw_computed = ds_chunk.HW.compute()

        # Apply the condition and compute the chunk
        ds_hw_chunk = ds_chunk.where(hw_computed).compute()
        ds_no_hw_chunk = ds_chunk.where(~hw_computed).compute()

        # # Append the processed chunk to the list
        # print(f"Appending HW to Zarr", ds_hw_chunk.time.values[0], ds_hw_chunk.time.values[-1])
        # append_to_zarr(ds_hw_chunk, os.path.join(zarr_path, 'HW'))
        # print(f"Appending No HW to Zarr", ds_no_hw_chunk.time.values[0], ds_no_hw_chunk.time.values[-1])
        # append_to_zarr(ds_no_hw_chunk, os.path.join(zarr_path, 'NO_HW'))
        
    return 
#%%
zarr_path = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/zarr2'

# Apply the function to your dataset
process_in_chunks(ds=ds, chunk_size=24 * 3 , zarr_path=zarr_path)  # Adjust chunk_size as needed

# Now ds_hw and ds_no_hw contain the processed data

#%%
