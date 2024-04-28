import os
import glob
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import yaml

import psutil


def show_memory_usage(message):
    # Get the current process
    process = psutil.Process(os.getpid())

    # Get the memory usage in bytes
    memory_info = process.memory_info()

    # Convert bytes to megabytes
    memory_usage_mb = memory_info.rss / (1024 * 1024)

    # Print the current memory usage
    print(f"{message} memory usage: {memory_usage_mb:.2f} MB")


def load_config(config_file):
    """Loads configuration parameters from a YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def extract_variables(input_pattern, variables, output_file, log_file_path):
    """Extracts specified variables from input files and saves them to an output file."""
    print(f"Extracting variables {variables} from {input_pattern} into {output_file}")
    file_paths = sorted(glob.glob(input_pattern))

    for file_path in file_paths:
        log_file_status(log_file_path, file_path, "Processed")

    ds = xr.open_mfdataset(file_paths, combine='by_coords')
    ds_var = ds[variables]
    ds_var.to_netcdf(output_file)


def convert_time(ds):
    """Converts time values in the dataset to pandas Timestamps."""
    ds['time'] = [pd.Timestamp(time.strftime()) for time in ds['time'].values]
    return ds


def round_to_nearest_hour(time_values):
    """Rounds cftime.DatetimeNoLeap objects to the nearest hour."""
    rounded_times = []
    for time_value in time_values:
        year, month, day, hour = time_value.year, time_value.month, time_value.day, time_value.hour
        minute = time_value.minute

        if minute >= 30 and hour < 23:
            hour += 1
        elif minute >= 30 and hour == 23:
            new_day = cftime.DatetimeNoLeap(year, month, day) + cftime.timedelta(days=1)
            year, month, day = new_day.year, new_day.month, new_day.day
            hour = 0

        rounded_time = cftime.DatetimeNoLeap(year, month, day, hour)
        rounded_times.append(rounded_time)

    return np.array(rounded_times)


def set_unwanted_to_nan(ds):
    """Sets unwanted data to NaN while keeping the dataset structure."""
    condition_jja_nh = (ds['time.season'] == 'JJA') & (ds['lat'] >= 0)
    condition_djf_sh = (ds['time.season'] == 'DJF') & (ds['lat'] < 0)
    condition_tsa_u_not_null = ds['TSA_U'].notnull()
    condition = (condition_jja_nh | condition_djf_sh) & condition_tsa_u_not_null
    ds_filtered = ds.where(condition)
    return ds_filtered


def log_file_status(log_file_path, file_path, status):
    """Logs the status of each file."""
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_path} - {status}\n')


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


def zarr_to_dataframe(zarr_path, start_year, end_year, years_per_chunk, output_path, hw_flag, core_vars=None):
    """Converts Zarr data to DataFrame format and saves it as Parquet files."""
    ds = xr.open_zarr(os.path.join(zarr_path, hw_flag), chunks='auto')
    if core_vars:
        ds = ds[core_vars]

    chunk_start_year = start_year
    while chunk_start_year <= end_year:
        df_list = []
        chunk_end_year = min(chunk_start_year + years_per_chunk - 1, end_year)

        for year in range(chunk_start_year, chunk_end_year + 1):
            ds_year = ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
            df_year = ds_year.to_dataframe(['lat', 'lon', 'time']).dropna()
            df_list.append(df_year)

        df_chunk = pd.concat(df_list)
        df_chunk.columns = df_chunk.columns.str.replace('UBWI', 'UWBI')

        parquet_file_path = os.path.join(output_path, f'ALL_{hw_flag}_{chunk_start_year}_{chunk_end_year}.parquet')
        df_chunk.to_parquet(parquet_file_path, engine='pyarrow', index=True)

        print(f'Saved {chunk_start_year}-{chunk_end_year} data to {parquet_file_path}')

        chunk_start_year = chunk_end_year + 1


def load_data_from_parquet(data_dir, start_year, end_year):
    """Load HW and NO_HW data from the specified directory for a range of years."""
    df_hw_list = []
    df_no_hw_list = []

    for year in range(start_year, end_year + 1):
        file_name_hw = os.path.join(data_dir, f"ALL_HW_{year}_{year}.parquet")
        file_name_no_hw = os.path.join(data_dir, f"ALL_NO_HW_{year}_{year}.parquet")

        df_hw_list.append(pd.read_parquet(file_name_hw))
        df_no_hw_list.append(pd.read_parquet(file_name_no_hw))

    return pd.concat(df_hw_list), pd.concat(df_no_hw_list)


def add_year_month_hour_cols(df):
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


def calculate_uhi_diff(df_hw, df_no_hw_avg):
    """Calculate the difference between UHI values of HW and average NO_HW on matching columns."""
    merged_df = pd.merge(df_hw, df_no_hw_avg, on=['lat', 'lon', 'year', 'hour'],
                         suffixes=('', '_avg'))
    merged_df['UHI_diff'] = merged_df['UHI'] - merged_df['UHI_avg']
    merged_df['UWBI_diff'] = merged_df['UWBI'] - merged_df['UWBI_avg']
    return merged_df


def convert_time_to_local_and_add_hour(df):
    """Adjusts DataFrame time data to local based on longitude and extracts local hour."""

    def calculate_timezone_offset(longitude):
        return np.floor(longitude / 15.0).astype(int)

    offsets = calculate_timezone_offset(df['lon'].values)
    df['local_time'] = df['time'] + pd.to_timedelta(offsets, unit='h')
    df['local_hour'] = df['local_time'].dt.hour
    return df


def add_event_id(df):
    """Add event_ID to the DataFrame."""
    df.sort_values(by=['location_ID', 'time'], inplace=True)
    df['time_diff'] = df.groupby('location_ID')['time'].diff().dt.total_seconds() / 3600
    df['new_event'] = (df['time_diff'] > 1)
    df['event_ID'] = df.groupby('location_ID')['new_event'].cumsum()
    df['global_event_ID'] = df['location_ID'].astype(str) + '_' + df['event_ID'].astype(str)
    return df


def main():
    """Main function that orchestrates the entire process."""
    # Load configuration from YAML
    config = load_config("config.yaml")

    show_memory_usage("Before run_extract_variables")
    # Extract variables from daily h1 files
    if config["run_extract_variables"]:
        extract_variables(config["daily_file_pattern"], config["daily_variables_list"],
                          config["daily_extracted_cols_file"], config["log_file_path"])

    show_memory_usage("Before run_hw_detection")
    # Determine urban grid and heatwave days
    if config["run_hw_detection"]:
        ds_one_monthly_data = xr.open_dataset(config["one_simu_result_monthly_file"])
        urban_non_null_mask = ds_one_monthly_data['TSA_U'].isel(time=0).notnull().drop('time')

        ds_hw = xr.open_dataset(config["daily_extracted_cols_file"])

        def detect_heatwave(tsa_r_np):
            tsa_r_np = np.atleast_1d(tsa_r_np)
            hw = np.full(tsa_r_np.shape, np.nan)
            kelvin_threshold = config["kelvin_threshold"]

            for i in range(2, len(tsa_r_np)):
                if (tsa_r_np[i - 2] > kelvin_threshold and
                        tsa_r_np[i - 1] > kelvin_threshold and
                        tsa_r_np[i] > kelvin_threshold):
                    hw[i - 2:i + 1] = 1

            return hw

        hw = xr.apply_ufunc(
            detect_heatwave, ds_hw['TREFMXAV_R'],
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True,
            output_dtypes=[bool]
        )

        ds_hw['HW'] = hw
        ds_hw_filtered = ds_hw.where(urban_non_null_mask.broadcast_like(ds_hw), drop=False)
        ds_hw_filtered.to_netcdf(config["daily_grid_hw_file"])

        # Determine heatwave dates
        daily_hw_urban_count = hw.sum(dim=['lat', 'lon']).compute()
        hw_dates = daily_hw_urban_count.where(daily_hw_urban_count > 1, drop=True)
        hw_dates.to_netcdf(config["daily_hw_dates_file"])

    # Convert NetCDF to Zarr
    if config["run_convert_to_zarr"]:
        one_hourly_file = config["one_hourly_file"]
        ds_one_hourly_data = xr.open_dataset(one_hourly_file)
        vars = [var for var in ds_one_hourly_data.data_vars if (len(ds_one_hourly_data[var].dims) == 3)]

        ds_daily_grid_hw = xr.open_dataset(config["daily_grid_hw_file"])
        ds_daily_grid_hw = convert_time(ds_daily_grid_hw)

        for year in range(config["start_year"], config["end_year"] + 1):
            netcdf_to_zarr_process_year(config["sim_results_hourly_dir"], config["research_results_zarr_dir"],
                                        config["log_file_path"], year, vars, ds_daily_grid_hw)

    # Separate HW and No HW data
    if config["run_sep_hw_no_hw"]:
        ds = xr.open_zarr(os.path.join(config["research_results_zarr_dir"], '3Dvars'))
        ds['UHI'] = ds.TSA_U - ds.TSA_R
        ds['UWBI'] = ds.WBA_U - ds.WBA_R
        separate_hw_no_hw_process_in_chunks(ds=ds, chunk_size=24 * 3, zarr_path=config["research_results_zarr_dir"])

    # Convert Zarr to Parquet
    if config["run_zarr_to_parquet"]:
        zarr_to_dataframe(config["research_results_zarr_dir"], config["start_year"], config["end_year"],
                          1, config["research_results_parquet_dir"], 'HW', config["core_vars"])
        zarr_to_dataframe(config["research_results_zarr_dir"], config["start_year"], config["end_year"],
                          1, config["research_results_parquet_dir"], 'NO_HW', config["core_vars"])

    # Main script
    df_hw, df_no_hw = load_data_from_parquet(config["research_results_parquet_dir"],
                                             config["start_year"], config["end_year"])

    # Prepare DataFrame by decomposing datetime
    df_hw = add_year_month_hour_cols(df_hw)
    df_no_hw = add_year_month_hour_cols(df_no_hw)

    df_no_hw_avg = df_no_hw.groupby(['lat', 'lon', 'year', 'hour']).mean()
    local_hour_adjusted_df = calculate_uhi_diff(df_hw, df_no_hw_avg)
    local_hour_adjusted_df = convert_time_to_local_and_add_hour(local_hour_adjusted_df)
    local_hour_adjusted_df.rename(columns=lambda x: x.replace('UBWI', 'UWBI'), inplace=True)

    # add location ID
    location_ds = xr.open_dataset(config["loc_id_path"])
    location_df = location_ds.to_dataframe().reset_index()

    # Merge the location_df with the local_hour_adjusted_df
    local_hour_adjusted_df = pd.merge(local_hour_adjusted_df, location_df, on=['lat', 'lon'], how='left')

    local_hour_adjusted_df = add_event_id(local_hour_adjusted_df)

    local_hour_adjusted_df.to_feather(config["var_with_id_path"])


if __name__ == "__main__":
    main()
