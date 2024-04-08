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


def extract_variables(input_pattern, variables, output_file, log_file_path):
    """
    Extracts specified variables from input files and saves them to an output file.
    """
    print(f"Extracting variables {variables} from {input_pattern} into {output_file}")
    file_paths = sorted(glob.glob(input_pattern))

    for file_path in file_paths:
        log_file_status(log_file_path, file_path, "Processed")

    ds = xr.open_mfdataset(file_paths, combine='by_coords')
    ds_var = ds[variables]
    ds_var.to_netcdf(output_file)


def convert_time(ds):
    """
    Converts time values in the dataset to pandas Timestamps.
    """
    ds['time'] = [pd.Timestamp(time.strftime()) for time in ds['time'].values]
    return ds


def round_to_nearest_hour(time_values):
    """
    Rounds cftime.DatetimeNoLeap objects to the nearest hour.
    """
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
    """
    Sets unwanted data to NaN while keeping the dataset structure.
    """
    condition_jja_nh = (ds['time.season'] == 'JJA') & (ds['lat'] >= 0)
    condition_djf_sh = (ds['time.season'] == 'DJF') & (ds['lat'] < 0)
    condition_tsa_u_not_null = ds['TSA_U'].notnull()
    condition = (condition_jja_nh | condition_djf_sh) & condition_tsa_u_not_null
    ds_filtered = ds.where(condition)
    return ds_filtered


def log_file_status(log_file_path, file_path, status):
    """
    Logs the status of each file.
    """
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_path} - {status}\n')


def append_to_zarr(ds, zarr_group):
    """
    Appends data to a Zarr group.
    """
    chunk_size = {'time': 24 * 3 * 31, 'lat': 96, 'lon': 144}
    ds = ds.chunk(chunk_size)
    if os.path.exists(zarr_group):
        ds.to_zarr(zarr_group, mode='a', append_dim='time', consolidated=True)
    else:
        encoding = {var: {'compressor': zarr.Blosc(cname='zstd', clevel=3)} for var in ds.data_vars}
        ds.to_zarr(zarr_group, mode='w', encoding=encoding, consolidated=True)


def netcdf_to_zarr_process_year(netcdf_dir, zarr_path, log_file_path, year, var_list, ds_daily_grid_hw):
    """

    Processes data for a specific year and appends it to the Zarr group.
    """
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
    #todo: HW here is not dependent if we have data in that cell. we could have an empty cell because we filtered out
    # the non-summer months
    ds_year['HW'] = hourly_mask_year

    append_to_zarr(ds_year, os.path.join(zarr_path, '3Dvars'))


def separate_hw_no_hw_process_in_chunks(ds, chunk_size, zarr_path):
    """
    Processes data in smaller chunks and separates HW and No HW data.
    """
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
    """
    Converts Zarr data to DataFrame format and saves it as Parquet files.
    """
    ds = xr.open_zarr(os.path.join(zarr_path, hw_flag), chunks='auto')
    # core_vars = ['UHI', 'UBWI']
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
        parquet_file_path = os.path.join(output_path, f'{hw_flag}_{chunk_start_year}_{chunk_end_year}.parquet')
        df_chunk.to_parquet(parquet_file_path, engine='pyarrow', index=True)

        print(f'Saved {chunk_start_year}-{chunk_end_year} data to {parquet_file_path}')

        chunk_start_year = chunk_end_year + 1


def main():
    """
    Main function that orchestrates the entire process.
    """
    case_name = 'i.e215.I2000Clm50SpGs.hw_production.02'
    case_results_dir = '/Trex/case_results'
    sim_results_hourly_dir = os.path.join(case_results_dir, case_name, 'sim_results/hourly')
    sim_results_daily_dir = os.path.join(case_results_dir, case_name, 'sim_results/daily')
    sim_results_monthly_dir = os.path.join(case_results_dir, case_name, 'sim_results/monthly')

    research_results_zarr_dir = os.path.join(case_results_dir, case_name, 'research_results/zarr')
    os.makedirs(research_results_zarr_dir, exist_ok=True)

    research_results_summary_dir = os.path.join(case_results_dir, case_name, 'research_results/summary')
    os.makedirs(research_results_summary_dir, exist_ok=True)
    log_file_path = os.path.join(research_results_summary_dir, 'processed_files.log')
    # process_data_dir = '/home/jguo/process_data'
    start_year = 1985
    end_year = 2014

    run_all = False
    run_extract_variables = False
    run_hw_detection = False
    run_convert_to_zarr = False
    run_sep_hw_no_hw = False
    run_zarr_to_parquet = True

    one_simu_result_monthly_file = os.path.join(sim_results_monthly_dir, f'{case_name}.clm2.h0.1985-01.nc')

    # Extract variables from daily h1 files
    daily_file_pattern = os.path.join(sim_results_daily_dir, f'{case_name}.clm2.h1.*-00000.nc')
    variables_list = ['TSA', 'TSA_U', 'TSA_R', 'TREFMXAV_R']
    daily_output_file = os.path.join(research_results_summary_dir, f'{case_name}.clm2.h1.TSA_UR_TREFMXAV_R.nc')
    if run_all or run_extract_variables:
        extract_variables(daily_file_pattern, variables_list, daily_output_file, log_file_path)

    if run_all or run_hw_detection:
        # Determine urban grid
        ds_one_monthly_data = xr.open_dataset(one_simu_result_monthly_file)
        urban_non_null_mask = ds_one_monthly_data['TSA_U'].isel(time=0).notnull().drop('time')

        # Determine heatwave days
        # hw_input_file_path = os.path.join(research_results_summary_dir, f'{case_name}.clm2.h1.TSA_UR_TREFMXAV_R.nc')
        hw_input_file_path = daily_output_file
        ds_hw = xr.open_dataset(hw_input_file_path)

        fahrenheit_threshold = 90
        kelvin_threshold = (fahrenheit_threshold - 32) * (5 / 9) + 273.15

        def detect_heatwave(tsa_r_np):
            tsa_r_np = np.atleast_1d(tsa_r_np)
            hw = np.full(tsa_r_np.shape, np.nan)

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
            output_dtypes=[float]
        )

        ds_hw['HW'] = hw
        # mask out non-urban grid cells
        ds_hw_filtered = ds_hw.where(urban_non_null_mask.broadcast_like(ds_hw), drop=False)
        ds_hw_filtered.to_netcdf(os.path.join(research_results_summary_dir, 'daily_grid_hw.nc'))

        # Determine heatwave dates
        daily_hw_urban_count = hw.sum(dim=['lat', 'lon']).compute()
        hw_dates = daily_hw_urban_count.where(daily_hw_urban_count > 1, drop=True)
        hw_dates.to_netcdf(os.path.join(research_results_summary_dir, 'daily_hw_dates.nc'))

    # Convert NetCDF to Zarr
    if run_all or run_convert_to_zarr:
        one_hourly_file = os.path.join(sim_results_hourly_dir, f'{case_name}.clm2.h2.1985-07-01-00000.nc')
        ds_one_hourly_data = xr.open_dataset(one_hourly_file)
        vars = [var for var in ds_one_hourly_data.data_vars if (len(ds_one_hourly_data[var].dims) == 3)]

        ds_daily_grid_hw = xr.open_dataset(os.path.join(research_results_summary_dir, 'daily_grid_hw.nc'))
        ds_daily_grid_hw = convert_time(ds_daily_grid_hw)

        for year in range(start_year, end_year + 1):
            netcdf_to_zarr_process_year(sim_results_hourly_dir, research_results_zarr_dir, log_file_path,
                                        year, vars, ds_daily_grid_hw)

    # Separate HW and No HW data
    if run_all or run_sep_hw_no_hw:
        ds = xr.open_zarr(os.path.join(research_results_zarr_dir, '3Dvars'))
        # ds = ds.sel(time=slice('1985-01-02', '1985-12-31'))
        ds['UHI'] = ds.TSA_U - ds.TSA_R
        ds['UWBI'] = ds.WBA_U - ds.WBA_R
        core_vars = ['TSA', 'TSA_R', 'TSA_U', 'WBA', 'WBA_R', 'WBA_U', 'HW']
        separate_hw_no_hw_process_in_chunks(ds=ds, chunk_size=24 * 3, zarr_path=research_results_zarr_dir)

    # Convert Zarr to Parquet
    if run_all or run_zarr_to_parquet:
        zarr_to_dataframe(research_results_zarr_dir, start_year, end_year, 10,
                          research_results_zarr_dir, 'HW')
        zarr_to_dataframe(research_results_zarr_dir, start_year, end_year, 10,
                          research_results_zarr_dir, 'NO_HW')


if __name__ == "__main__":
    main()
