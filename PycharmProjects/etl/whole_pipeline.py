import os
import glob
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import zarr



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
            #this the key line the convert columnar data to row based data format
            df_year = ds_year.to_dataframe(['lat', 'lon', 'time']).dropna()
            df_list.append(df_year)

        df_chunk = pd.concat(df_list)
        # correct the typo in the column name in zarr files
        df_chunk.columns = df_chunk.columns.str.replace('UBWI', 'UWBI')

        parquet_file_path = os.path.join(output_path, f'ALL_{hw_flag}_{chunk_start_year}_{chunk_end_year}.parquet')
        df_chunk.to_parquet(parquet_file_path, engine='pyarrow', index=True)

        print(f'Saved {chunk_start_year}-{chunk_end_year} data to {parquet_file_path}')

        chunk_start_year = chunk_end_year + 1


def load_data_from_parquet(data_dir, start_year, end_year):
    """
    Load HW and NO_HW data from the specified directory for a range of years.

    Args:
        data_dir (str): Directory where `.parquet` files are stored.
        start_year (int): Starting year.
        end_year (int): Ending year.

    Returns:
        tuple of pd.DataFrame: Returns two DataFrames, one for HW events and one for NO_HW events.
    """
    df_hw_list = []
    df_no_hw_list = []

    # Iterate through each year and collect data
    for year in range(start_year, end_year + 1):
        file_name_hw = os.path.join(data_dir, f"ALL_HW_{year}_{year}.parquet")
        file_name_no_hw = os.path.join(data_dir, f"ALL_NO_HW_{year}_{year}.parquet")

        df_hw_list.append(pd.read_parquet(file_name_hw))
        df_no_hw_list.append(pd.read_parquet(file_name_no_hw))

    return pd.concat(df_hw_list), pd.concat(df_no_hw_list)


def add_year_month_hour_cols(df):
    """
    Decompose the 'time' index of DataFrame into 'hour', 'month', and 'year' and append them as columns.

    Args:
        df (pd.DataFrame): DataFrame whose time index needs to be decomposed.

    Returns:
        pd.DataFrame: DataFrame with added 'hour', 'month', 'year' columns.
    """
    df['hour'] = df.index.get_level_values('time').hour
    df['month'] = df.index.get_level_values('time').month
    df['year'] = df.index.get_level_values('time').year
    return df


def check_data_overlap(df_hw, df_no_hw):
    """
    Check if there is any overlap in MultiIndexes between two DataFrames.

    Args:
        df_hw (pd.DataFrame): DataFrame containing HW data.
        df_no_hw (pd.DataFrame): DataFrame containing NO_HW data.

    Returns:
        set: Set of overlapping indices, if any.
    """
    keys_hw = set(df_hw.index)
    keys_no_hw = set(df_no_hw.index)

    return keys_hw & keys_no_hw


def calculate_uhi_diff(df_hw, df_no_hw_avg):
    """
    Calculate the difference between UHI values of HW and average NO_HW on matching columns.

    Args:
        df_hw (pd.DataFrame): DataFrame containing HW data.
        df_no_hw_avg (pd.DataFrame): DataFrame containing averaged NO_HW data.

    Returns:
        pd.DataFrame: DataFrame with added 'UHI_diff' and 'UBWI_diff' columns.
    """
    merged_df = pd.merge(df_hw, df_no_hw_avg, on=['lat', 'lon', 'year', 'hour'],
                         suffixes=('', '_avg'))
    merged_df['UHI_diff'] = merged_df['UHI'] - merged_df['UHI_avg']
    merged_df['UWBI_diff'] = merged_df['UWBI'] - merged_df['UWBI_avg']
    return merged_df


def convert_time_to_local_and_add_hour(df):
    """
    Adjusts DataFrame time data to local based on longitude and extracts local hour.

    Args:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: Modified DataFrame with 'local_time' and 'local_hour' columns.
    """

    def calculate_timezone_offset(longitude):
        return np.floor(longitude / 15.0).astype(int)

    offsets = calculate_timezone_offset(df['lon'].values)
    df['local_time'] = df['time'] + pd.to_timedelta(offsets, unit='h')
    df['local_hour'] = df['local_time'].dt.hour
    return df


def add_event_id(df):
    """
    Add event_ID to the DataFrame. This only depends on cell location_id and date.
    It is only for HW dataset hence a three column dataframe with location_id, date,
    event_ID can be created independently and merged with any HW dataframe.

    Args:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: Modified DataFrame with 'event_ID' and 'global_event_ID' columns.
    """
    # Sort by 'location_ID' and 'time'
    df.sort_values(by=['location_ID', 'time'], inplace=True)

    # Create a new column 'time_diff' to find the difference in hours between consecutive rows
    df['time_diff'] = df.groupby('location_ID')['time'].diff().dt.total_seconds() / 3600

    # Identify the start of a new event (any gap of more than one hour)
    df['new_event'] = (df['time_diff'] > 1)

    # Generate cumulative sum to assign unique event IDs within each location
    df['event_ID'] = df.groupby('location_ID')['new_event'].cumsum()

    # Combine location_ID with event_ID to create a globally unique event identifier
    df['global_event_ID'] = df['location_ID'].astype(str) + '_' + df['event_ID'].astype(str)

    return df


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

    research_results_parquet_dir = os.path.join(case_results_dir, case_name, 'research_results/parquet')
    os.makedirs(research_results_parquet_dir, exist_ok=True)

    log_file_path = os.path.join(research_results_summary_dir, 'processed_files.log')
    # process_data_dir = '/home/jguo/process_data'
    start_year = 1985
    end_year = 2013

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
        separate_hw_no_hw_process_in_chunks(ds=ds, chunk_size=24 * 3, zarr_path=research_results_zarr_dir)

    # Convert Zarr to Parquet
    if run_all or run_zarr_to_parquet:
        # core_vars = ['UHI', 'UBWI', 'WIND', 'HW', 'APPAR_TEMP_R', 'APPAR_TEMP_U', 'EFLX_LH_TOT_R', 'EFLX_LH_TOT_U',
        #              'FGR_R', 'FGR_U', 'FIRA_R', 'FIRA_U', 'FIRE_R', 'FIRE_U', 'FSA_R', 'FSA_U',
        #              'FSH_R', 'FSH_U', 'HIA_R', 'HIA_U', 'Q2M_R', 'Q2M_U', 'TSA_R', 'TSA_U',
        #              'TSKIN_R', 'TSKIN_U', 'VAPOR_PRES_R', 'VAPOR_PRES_U', 'WBA_R', 'WBA_U']
        core_vars = ['UHI', 'UBWI', 'WIND', 'RAIN', 'SNOW', 'HW', 'Q2M_R', 'Q2M_U', 'VAPOR_PRES_R', 'VAPOR_PRES_U']
        core_vars = ['TSA', 'TSA_R', 'TSA_U', 'Q2M', 'Q2M_U', 'Q2M_R', 'WBA_U', 'WBA_R', 'WBA', 'VAPOR_PRES',
                     'VAPOR_PRES_U',
                     'VAPOR_PRES_R', 'WASTEHEAT', 'HEAT_FROM_AC', 'URBAN_HEAT', 'FSDS', 'FLDS', 'FIRE', 'FIRE_U',
                     'FIRE_R',
                     'FIRA', 'FIRA_U', 'FIRA_R', 'FSA', 'FSA_U', 'FSA_R', 'EFLX_LH_TOT', 'EFLX_LH_TOT_R',
                     'EFLX_LH_TOT_U',
                     'FSH_R', 'FSH_U', 'FSH', 'FGR_R', 'FGR_U', 'FGR', 'RAIN', 'SNOW', 'TBOT', 'QBOT', 'PBOT', 'WIND',
                     'THBOT',
                     'TSKIN', 'TSKIN_U', 'TSKIN_R', 'APPAR_TEMP', 'APPAR_TEMP_U', 'APPAR_TEMP_R', 'HIA', 'HIA_U',
                     'HIA_R', 'U10']
        zarr_to_dataframe(research_results_zarr_dir, start_year, end_year, 1,
                          research_results_parquet_dir, 'HW', core_vars)
        zarr_to_dataframe(research_results_zarr_dir, start_year, end_year, 1,
                          research_results_parquet_dir, 'NO_HW', core_vars)

    # Main script
    # research_results_parquet_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/parquet'
    # research_results_summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'
    # start_year = 1985
    # end_year = 2013

    df_hw, df_no_hw = load_data_from_parquet(research_results_parquet_dir, start_year, end_year)

    # Prepare DataFrame by decomposing datetime
    df_hw = add_year_month_hour_cols(df_hw)
    df_no_hw = add_year_month_hour_cols(df_no_hw)

    # overlap = check_data_overlap(df_hw, df_no_hw)
    # if not overlap:
    #     print("There is no overlap between the keys of df_hw and df_no_hw.")
    # else:
    #     print("The following keys overlap between df_hw and df_no_hw:", overlap)

    df_no_hw_avg = df_no_hw.groupby(['lat', 'lon', 'year', 'hour']).mean()
    local_hour_adjusted_df = calculate_uhi_diff(df_hw, df_no_hw_avg)
    local_hour_adjusted_df = convert_time_to_local_and_add_hour(local_hour_adjusted_df)
    local_hour_adjusted_df.rename(columns=lambda x: x.replace('UBWI', 'UWBI'), inplace=True)

    # add location ID
    # location_id is only dependent on lat and lon.
    # Hence a three columns dataframe with lat, lon and location_id can be created independently
    # Load the NetCDF file

    loc_id_path = os.path.join(research_results_summary_dir, 'location_IDs.nc')
    location_ds = xr.open_dataset(loc_id_path)
    location_df = location_ds.to_dataframe().reset_index()

    # Merge the location_df with the local_hour_adjusted_df
    local_hour_adjusted_df = pd.merge(local_hour_adjusted_df, location_df, on=['lat', 'lon'], how='left')

    local_hour_adjusted_df = add_event_id(local_hour_adjusted_df)

    # merged_feather_path = os.path.join(research_results_summary_dir, 'local_hour_adjusted_variables.feather')
    var_with_id_path = os.path.join(research_results_summary_dir,
                                    'local_hour_adjusted_variables_with_location_ID.feather')
    local_hour_adjusted_df.to_feather(var_with_id_path)
