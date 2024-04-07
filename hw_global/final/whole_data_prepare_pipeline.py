import os
import glob
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import cartopy.crs as ccrs

def convert_time(ds):
    """Converts time values in the dataset to pandas Timestamps."""
    ds['time'] = [pd.Timestamp(time.strftime()) for time in ds['time'].values]
    return ds

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

def process_year(netcdf_dir, zarr_path, log_file_path, year, var_list, ds_daily_grid_hw):
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

    # Create a daily mask for the period where HW == 1 for the current year
    overlapping_mask_year = (
            ds_daily_grid_hw_year['HW'].sel(time=slice(ds_year['time'].min(), ds_year['time'].max())) == 1)

    # Expand the daily mask to hourly using forward fill for the current year
    full_hourly_range_year = pd.date_range(start=ds_year['time'].min().values, end=ds_year['time'].max().values,
                                           freq='H')
    hourly_mask_year = overlapping_mask_year.reindex(time=full_hourly_range_year, method='ffill').compute()

    # Apply the hourly mask to ds_year to get filtered data for the current year
    ds_year['HW'] = hourly_mask_year

    append_to_zarr(ds_year, os.path.join(zarr_path, '3Dvars'))

def process_in_chunks(ds, chunk_size, zarr_path):
    # Determine the number of time steps
    num_time_steps = ds.dims['time']

    # Iterate over the dataset in chunks
    for start in range(0, num_time_steps, chunk_size):
        end = start + chunk_size
        print(f"Processing time steps {start} to {min(end, num_time_steps)}")

        # Select the chunk
        ds_chunk = ds.isel(time=slice(start, end))

        # Compute the boolean indexer for the current chunk
        hw_computed = ds_chunk.HW.compute()

        # Apply the condition and compute the chunk
        ds_hw_chunk = ds_chunk.where(hw_computed).compute()
        ds_no_hw_chunk = ds_chunk.where(~hw_computed).compute()

        # Append the processed chunk to the list
        print(f"Appending HW to Zarr", ds_hw_chunk.time.values[0], ds_hw_chunk.time.values[-1])
        append_to_zarr(ds_hw_chunk, os.path.join(zarr_path, 'HW'))
        print(f"Appending No HW to Zarr", ds_no_hw_chunk.time.values[0], ds_no_hw_chunk.time.values[-1])
        append_to_zarr(ds_no_hw_chunk, os.path.join(zarr_path, 'NO_HW'))

def zarr_to_dataframe(zarr_path, start_year, end_year, years_per_chunk, output_path, hw_flag):
    # Read the Zarr dataset using Xarray with automatic chunking for Dask
    ds = xr.open_zarr(os.path.join(zarr_path, hw_flag), chunks='auto')

    # Select core variables
    core_vars = ['UHI', 'UBWI']
    ds = ds[core_vars]

    for start_chunk_year in range(start_year, end_year + 1, years_per_chunk):
        # Initialize an empty list to hold DataFrames for each year
        df_list = []

        # Determine the end year for the current chunk, ensuring it does not exceed the end_year
        end_chunk_year = min(start_chunk_year + years_per_chunk - 1, end_year)

        for year in range(start_chunk_year, end_chunk_year + 1):
            # Select the data for the current year
            ds_year = ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))

            # Convert to DataFrame without resetting the index
            df_year = ds_year.to_dataframe(['lat', 'lon', 'time']).dropna()

            # Append the DataFrame for the current year to the list
            df_list.append(df_year)

        # Concatenate all DataFrames in the list to create a single DataFrame for the 10-year chunk
        df_chunk = pd.concat(df_list)

        # Define the path to the Parquet file for this 10-year chunk
        parquet_file_path = os.path.join(output_path, f'{hw_flag}_{start_chunk_year}_{end_chunk_year}.parquet')

        # Write the 10-year chunk DataFrame to a Parquet file
        df_chunk.to_parquet(parquet_file_path, engine='pyarrow', index=True)

        print(f'Saved {start_chunk_year}-{end_chunk_year} data to {parquet_file_path}')

def main():
    monthly_file = '/tmpdata/summerized_data/i.e215.I2000Clm50SpGs.hw_production.02/sub_sample/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h0.1985-01.nc'
    hw_summary_dir = '/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/summary'
    netcdf_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/sim_results/hourly'
    zarr_path = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/zarr'
    output_dir = zarr_path
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'processed_files.log')

    # Determine urban grid
    ds_monthly = xr.open_dataset(monthly_file)
    urban_non_null_mask = ds_monthly['TSA_U'].isel(time=0).notnull().drop('time')

    # Determine heatwave days
    hw_input_file = 'i.e215.I2000Clm50SpGs.hw_production.02.clm2.h1.TSA_UR_TREFMXAV_R.nc'
    hw_input_file_path = os.path.join(hw_summary_dir, hw_input_file)
    ds_hw = xr.open_dataset(hw_input_file_path)

    fahrenheit_threshold = 90
    kelvin_threshold = (fahrenheit_threshold - 32) * (5/9) + 273.15

    def detect_heatwave(tsa_r_np):
        tsa_r_np = np.atleast_1d(tsa_r_np)
        hw = np.full(tsa_r_np.shape, np.nan)

        for i in range(2, len(tsa_r_np)):
            if (tsa_r_np[i-2] > kelvin_threshold and
                    tsa_r_np[i-1] > kelvin_threshold and
                    tsa_r_np[i] > kelvin_threshold):
                hw[i-2:i+1] = 1

        return hw

    hw = xr.apply_ufunc(
        detect_heatwave, ds_hw['TREFMXAV_R'],
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        output_dtypes=[float]
    )

    ds_hw['HW'] = hw
    ds_hw_filtered = ds_hw.where(urban_non_null_mask.broadcast_like(ds_hw), drop=False)
    ds_hw_filtered.to_netcdf(os.path.join(hw_summary_dir, 'daily_grid_hw.nc'))

    daily_hw_urban_count = hw.sum(dim=['lat', 'lon']).compute()
    hw_dates = daily_hw_urban_count.where(daily_hw_urban_count > 1, drop=True)
    hw_dates.to_netcdf(os.path.join(hw_summary_dir, 'daily_hw_dates.nc'))

    # Convert NetCDF to Zarr
    one_hourly_file = '/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/hourly_raw/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.1985-07-01-00000.nc'
    ds_hourly = xr.open_dataset(one_hourly_file)
    vars = [var for var in ds_hourly.data_vars if (len(ds_hourly[var].dims) == 3)]

    start_year = 1985
    end_year = 2014

    # Read in the daily grid hw netcdf file
    ds_daily_grid_hw = xr.open_dataset(os.path.join(hw_summary_dir, 'daily_grid_hw.nc'))
    ds_daily_grid_hw = convert_time(ds_daily_grid_hw)

    for year in range(start_year, end_year + 1):
        process_year(netcdf_dir, zarr_path, log_file_path, year, vars, ds_daily_grid_hw)

    # Separate HW and No HW data
    ds = xr.open_zarr(os.path.join(zarr_path, '3Dvars'))
    core_vars = ['TSA', 'TSA_R', 'TSA_U', 'WBA', 'WBA_R', 'WBA_U', 'HW']
    ds = ds.sel(time=slice('1985-01-02', '1985-12-31'))

    ds['UHI'] = ds.TSA_U - ds.TSA_R
    ds['UBWI'] = ds.WBA_U - ds.WBA_R

    process_in_chunks(ds=ds, chunk_size=24 * 3, zarr_path=zarr_path)

    # Convert Zarr to Parquet
    zarr_to_dataframe(zarr_path, start_year, end_year, 10, output_dir, 'HW')
    zarr_to_dataframe(zarr_path, start_year, end_year, 10, output_dir, 'NO_HW')

if __name__ == "__main__":
    main()