import os
import glob
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import zarr

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


def process_year(netcdf_dir, zarr_path, log_file_path, year, var_list, ds_daily_grid_hw):
    file_pattern = os.path.join(netcdf_dir, f'i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.{year}-*-00000.nc')
    file_paths = sorted(glob.glob(file_pattern))

    print("Processing", file_pattern)

    if not file_paths:
        log_file_status(log_file_path, f'No files found for {file_pattern}', "Missing")
        return

    ds_year = xr.open_mfdataset(file_paths, chunks={'time': 24 * 31})[var_list]
    ds_daily_grid_hw_year = ds_daily_grid_hw.sel(time=ds_daily_grid_hw['time'].dt.year == year).compute()

    # Store data types before modification
    data_types_before = {var: ds_year[var].dtype for var in ds_year.data_vars}
    print("Data types before set_unwanted_to_nan:")
    for var, dtype in data_types_before.items():
        print(f'{var}: {dtype}')

    ds_year['time'] = round_to_nearest_hour(ds_year['time'].values)
    ds_year = convert_time(ds_year)
    ds_year = set_unwanted_to_nan(ds_year)

    # Store data types after modification
    data_types_after = {var: ds_year[var].dtype for var in ds_year.data_vars}
    print("Data types after set_unwanted_to_nan:")
    for var, dtype in data_types_after.items():
        print(f'{var}: {dtype}')

    # Compare data types before and after
    print("\nData type changes:")
    for var in ds_year.data_vars:
        if data_types_before[var] != data_types_after[var]:
            print(f'{var}: {data_types_before[var]} -> {data_types_after[var]}')

    # Create a daily mask for the period where HW == 1 for the current year
    overlapping_mask_year = (
                ds_daily_grid_hw_year['HW'].sel(time=slice(ds_year['time'].min(), ds_year['time'].max())) == 1)

    # Expand the daily mask to hourly using forward fill for the current year
    full_hourly_range_year = pd.date_range(start=ds_year['time'].min().values, end=ds_year['time'].max().values,
                                           freq='H')
    hourly_mask_year = overlapping_mask_year.reindex(time=full_hourly_range_year, method='ffill').compute()

    # Apply the hourly mask to ds_year to get filtered data for the current year
    #todo: HW here is not dependent if we have data in that cell. we could have an empty cell because we filtered out
    # the non-summer months
    ds_year['HW'] = hourly_mask_year

    append_to_zarr(ds_year, os.path.join(zarr_path, '3Dvars'))


if __name__ == "__main__":
    # netcdf_dir = '/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/hourly_raw'
    netcdf_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/sim_results/hourly'
    zarr_path = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/zarr'
    output_dir = zarr_path
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'processed_files.log')

    one_hourly_file = '/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/hourly_raw/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.1985-07-01-00000.nc'
    ds_hourly = xr.open_dataset(one_hourly_file)
    vars = [var for var in ds_hourly.data_vars if (len(ds_hourly[var].dims) == 3)]

    start_year = 1985
    end_year = 2014


    # Read in the daily grid hw netcdf file
    ds_daily_grid_hw = xr.open_dataset('/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/summary/daily_grid_hw.nc')

    ds_daily_grid_hw = convert_time(ds_daily_grid_hw)


    for year in range(start_year, end_year + 1):
        process_year(netcdf_dir, zarr_path, log_file_path, year, vars, ds_daily_grid_hw)
