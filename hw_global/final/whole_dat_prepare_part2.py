import pandas as pd
import numpy as np
import os

def load_data(data_dir, start_year, end_year):
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

def calculate_difference(df_hw, df_no_hw_avg):
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

# Main script
data_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/parquet'
summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'
start_year = 1985
end_year = 2013

df_hw, df_no_hw = load_data(data_dir, start_year, end_year)

# Prepare DataFrame by decomposing datetime
df_hw = add_year_month_hour_cols(df_hw)
df_no_hw = add_year_month_hour_cols(df_no_hw)

# overlap = check_data_overlap(df_hw, df_no_hw)
# if not overlap:
#     print("There is no overlap between the keys of df_hw and df_no_hw.")
# else:
#     print("The following keys overlap between df_hw and df_no_hw:", overlap)

df_no_hw_avg = df_no_hw.groupby(['lat', 'lon', 'year', 'hour']).mean()
merged_df = calculate_difference(df_hw, df_no_hw_avg)
merged_df = convert_time_to_local_and_add_hour(merged_df)


merged_feather_path = os.path.join(summary_dir, 'local_hour_adjusted_variables.feather')
merged_df.to_feather(merged_feather_path)