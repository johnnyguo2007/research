import pandas as pd
import numpy as np
import xarray as xr
import os



def load_data(data_dir, start_year, end_year, columns=None):
    """
    Load HW and NO_HW data from the specified directory for a range of years.

    Args:
        data_dir (str): Directory where `.parquet` files are stored.
        start_year (int): Starting year.
        end_year (int): Ending year.
        columns (list): List of column names to read. If None, all columns are read.

    Returns:
        tuple of pd.DataFrame: Returns two DataFrames, one for HW events and one for NO_HW events.
    """
    df_hw_list = []
    df_no_hw_list = []

    # Ensure 'lon', 'lat', and 'time' are always included in the columns to read
    if columns is not None:
        columns = list(set(['lon', 'lat', 'time'] + columns))

    # Iterate through each year and collect data
    for year in range(start_year, end_year + 1):
        file_name_hw = os.path.join(data_dir, f"ALL_HW_{year}_{year}.parquet")
        file_name_no_hw = os.path.join(data_dir, f"ALL_NO_HW_{year}_{year}.parquet")

        # Read only specified columns
        df_hw_list.append(pd.read_parquet(file_name_hw, columns=columns))
        df_no_hw_list.append(pd.read_parquet(file_name_no_hw, columns=columns))

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



# Main script
data_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/parquet'
summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'
start_year = 1985
end_year = 1986


variables = {
    'Q2M': 'Global mean 2m Specific Humidity (Q2M)', 
    'FSH': 'Sensible Heat Flux (FSH)',
    'EFLX_LH_TOT': 'Latent Heat Flux (EFLX_LH_TOT)',
    'FSA': 'Shortwave Radiation (FSA)',
    'U10': 'Wind Speed (U10)',
}

df_hw, df_no_hw = load_data(data_dir, start_year, end_year, variables.keys())

# Prepare DataFrame by decomposing datetime
df_hw = add_year_month_hour_cols(df_hw)
df_no_hw = add_year_month_hour_cols(df_no_hw)

# overlap = check_data_overlap(df_hw, df_no_hw)
# if not overlap:
#     print("There is no overlap between the keys of df_hw and df_no_hw.")
# else:
#     print("The following keys overlap between df_hw and df_no_hw:", overlap)

# df_no_hw_avg = df_no_hw.groupby(['lat', 'lon', 'year', 'hour']).mean()
# local_hour_adjusted_df = calculate_uhi_diff(df_hw, df_no_hw_avg)
hw_local_hour_adjusted_df = convert_time_to_local_and_add_hour(df_hw)
no_hw_local_hour_adjusted_df = convert_time_to_local_and_add_hour(df_no_hw)
# local_hour_adjusted_df.rename(columns=lambda x: x.replace('UBWI', 'UWBI'), inplace=True)

# add location ID
# location_id is only dependent on lat and lon.
# Hence a three columns dataframe with lat, lon and location_id can be created independently
# Load the NetCDF file

loc_id_path = os.path.join(summary_dir, 'location_IDs.nc')
location_ds = xr.open_dataset(loc_id_path)
location_df = location_ds.to_dataframe().reset_index()

# Merge the location_df with the local_hour_adjusted_df
hw_local_hour_adjusted_df = pd.merge(hw_local_hour_adjusted_df, location_df, on=['lat', 'lon'], how='left')
hw_local_hour_adjusted_df = add_event_id(hw_local_hour_adjusted_df)

no_hw_local_hour_adjusted_df = pd.merge(no_hw_local_hour_adjusted_df, location_df, on=['lat', 'lon'], how='left')


# merged_feather_path = os.path.join(summary_dir, 'local_hour_adjusted_variables.feather')
var_with_id_path = os.path.join(summary_dir, 'investigate_FSH_hw.feather')
hw_local_hour_adjusted_df.to_feather(var_with_id_path)

var_with_id_path = os.path.join(summary_dir, 'investigate_FSH_no_hw.feather')
no_hw_local_hour_adjusted_df.to_feather(var_with_id_path)

