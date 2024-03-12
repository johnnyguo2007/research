import zarr
import os
import pandas as pd
import xarray as xr
from dask.distributed import Client

def convert_time(ds):
    """Converts time values in the dataset to pandas Timestamps."""
    ds['time'] = [pd.Timestamp(time.strftime()) for time in ds['time'].values]
    return ds

def append_to_zarr(ds, zarr_group):
    """Appends data to a Zarr dataset or creates a new one if it doesn't exist."""
    chunk_size = {'time': 24 * 3 * 31, 'lat': 96, 'lon': 144}
    ds = ds.chunk(chunk_size)
    if os.path.exists(zarr_group):
        ds.to_zarr(zarr_group, mode='a', append_dim='time', consolidated=True)
    else:
        encoding = {var: {'compressor': zarr.Blosc(cname='zstd', clevel=3)} for var in ds.data_vars}
        ds.to_zarr(zarr_group, mode='w', encoding=encoding, consolidated=True)

def main():
    # Initialize a Dask client to use multiple CPUs

    # Read the Zarr dataset using Xarray with automatic chunking for Dask
    ds = xr.open_zarr('/home/jguo/process_data/zarr/test02/3Dvars') #, chunks='auto')
    core_vars = ['TSA', 'TSA_R', 'TSA_U', 'WBA', 'WBA_R', 'WBA_U']
    ds = ds[core_vars]

    # Compute UHI and UWBI
    ds['UHI'] = ds['TSA_U'] - ds['TSA_R']
    ds['UWBI'] = ds['WBA_U'] - ds['WBA_R']

    # Read in the daily grid hw netcdf file
    ds_daily_grid_hw = xr.open_dataset('/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/summary/daily_grid_hw.nc')

    ds_daily_grid_hw = convert_time(ds_daily_grid_hw)
    ds = convert_time(ds)

    # Identify the range of years in the dataset
    start_year = ds['time'].dt.year.min().values
    end_year = ds['time'].dt.year.max().values

    for year in range(start_year, end_year + 1):
        print(f"Processing data for year: {year}")

        # Filter the dataset for the current year
        ds_year = ds.sel(time=ds['time'].dt.year == year).compute()
        ds_daily_grid_hw_year = ds_daily_grid_hw.sel(time=ds_daily_grid_hw['time'].dt.year == year).compute()

        # Example processing for each year
        # Note: Adjust the processing logic below as per your requirements

        # Create a daily mask for the period where HW == 1 for the current year
        overlapping_mask_year = (ds_daily_grid_hw_year['HW'].sel(time=slice(ds_year['time'].min(), ds_year['time'].max())) == 1)

        # Expand the daily mask to hourly using forward fill for the current year
        full_hourly_range_year = pd.date_range(start=ds_year['time'].min().values, end=ds_year['time'].max().values, freq='H')
        hourly_mask_year = overlapping_mask_year.reindex(time=full_hourly_range_year, method='ffill').compute()

        # Apply the hourly mask to ds_year to get filtered data for the current year
        filtered_ds_year = ds_year.where(hourly_mask_year)

        # Compute and save filtered datasets for the current year
        filtered_ds_compute_year = filtered_ds_year.compute()
        append_to_zarr(filtered_ds_compute_year, f'/home/jguo/process_data/zarr/test02/filtered_ds')
        del filtered_ds_compute_year

        # Invert the hourly mask to target hours where the condition doesn't hold for the current year
        inverted_hourly_mask_year = (~hourly_mask_year).compute()

        # Apply the inverted mask to ds_year to get unmasked data for the current year
        unmasked_ds_year = ds_year.where(inverted_hourly_mask_year)
        unmasked_ds_compute_year = unmasked_ds_year.compute()
        append_to_zarr(unmasked_ds_compute_year, f'/home/jguo/process_data/zarr/test02/unmasked_ds_year')
        del unmasked_ds_compute_year

if __name__ == "__main__":
    main()
