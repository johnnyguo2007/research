#%%
import pandas as pd
import numpy as np
import xarray as xr
import os
#%% md
# #  Load  and prepare HW and NO_HW df
#%% md
# ##  Load HW and NO_HW data
#%%
# Load the parquet data from /Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/parquet
#data_dir = '/Users/yguo/DataSpellProjects/hw/data/parquet'
data_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/parquet'
summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'

# Range of years for your data
start_year = 1985
end_year = 2013

# Initialize empty DataFrames to store the combined data
df_hw = pd.DataFrame()
df_no_hw= pd.DataFrame()

# Initialize empty lists to store DataFrames
df_hw_list = []
df_no_hw_list = []

# Iterate through each year and combine the Parquet files
for year in range(start_year, end_year + 1):
    # Construct file names for HW and NO_HW data
    file_name_hw = os.path.join(data_dir, f"ALL_HW_{year}_{year}.parquet")
    file_name_no_hw = os.path.join(data_dir, f"ALL_NO_HW_{year}_{year}.parquet")

    # Read Parquet files for the current year
    df_hw_tmp = pd.read_parquet(file_name_hw)
    df_no_hw_tmp = pd.read_parquet(file_name_no_hw)
    # 
    # df_hw_tmp = pd.read_parquet(file_name_hw, engine='pyarrow', dtype_backend='pyarrow')
    # df_no_hw_tmp = pd.read_parquet(file_name_no_hw, engine='pyarrow', dtype_backend='pyarrow')
    
    # Append DataFrames to lists
    df_hw_list.append(df_hw_tmp)
    df_no_hw_list.append(df_no_hw_tmp)

# Concatenate DataFrames from lists once outside the loop
df_hw = pd.concat(df_hw_list)
df_no_hw = pd.concat(df_no_hw_list)

# Now you have df_hw_all and df_no_hw_all containing the combined data from all years
print(df_hw.info())
print(df_no_hw.info())
#%%
df_hw.head()

#%% md
# ##  Add hour, month and year to the df_hw
# 
#%%
# Ensure 'time' is of datetime type
df_hw.index = df_hw.index.set_levels([df_hw.index.levels[0], df_hw.index.levels[1], pd.to_datetime(df_hw.index.levels[2])])

# Extract hour, month and year from 'time'
df_hw['hour'] = df_hw.index.get_level_values('time').hour
df_hw['month'] = df_hw.index.get_level_values('time').month
df_hw['year'] = df_hw.index.get_level_values('time').year
df_hw
#%%
# # Group by 'lat', 'lon', 'year', 'month', and 'hour', then calculate the mean of 'UHI' and 'UBWI'
# df_hw_avg = df_hw.groupby(['lat', 'lon', 'year', 'month', 'hour']).mean()
# df_hw_avg
#%%

# file_name_no_hw = 'NO_HW_1985_1994.parquet'
# #join data_dir and file_name
# no_hw_path = os.path.join(data_dir, file_name_no_hw)
# df_no_hw = pd.read_parquet(no_hw_path)
# print(df_no_hw.info())
# df_no_hw
#%%
#todo: add UHI and NO_HW UHI to make sure they are the same as the oringal netcdf data

#%% md
# ##  Validate there is not overlap between the HW and NO_HW data
#%%
#the key for both df_hw and df_no_hw are lat, lon and time. please show python code that they don't overlap on those keys
# Convert the MultiIndex of both DataFrames to sets
# keys_hw = set(df_hw.index)
# keys_no_hw = set(df_no_hw.index)
# 
# # Check if the intersection of these sets is empty
# overlap = keys_hw & keys_no_hw
# 
# # If the intersection is empty, print that there is no overlap. Otherwise, print the overlapping keys.
# if not overlap:
#     print("There is no overlap between the keys of df_hw and df_no_hw.")
# else:
#     print("The following keys overlap between df_hw and df_no_hw:")
#     print(overlap)
#%%

# group df_no_hw by lat, lon, year and hour of the day avaerage UHI and UBWI
df_no_hw.index = df_no_hw.index.set_levels(
    [df_no_hw.index.levels[0], df_no_hw.index.levels[1], pd.to_datetime(df_no_hw.index.levels[2])])
df_no_hw['hour'] = df_no_hw.index.get_level_values('time').hour
df_no_hw['year'] = df_no_hw.index.get_level_values('time').year
df_no_hw_avg = df_no_hw.groupby(['lat', 'lon', 'year', 'hour']).mean()
df_no_hw_avg
#%% md
# #  2: Calculate the difference between UHI in df_hw and df_no_hw
#%% md
# ##  UHI HW - NO_HW ( HW hour data - NO_HW yearl average data for the hour) 
# the df_no_hw_avg is the average value for a given hour of the day throughout the year.
# In the 2018 Zhao paper they seem to just do average the whole 30 years. 
# I want to substract the average UHI on the given hour for a given year from the hourly UHI data I have in df_hw, matching the year and hour between the two dataframes.
#%% md
# 
# ##  Step 2.1: Reset the index of df_hw and df_no_hw_avg (be careful on the increased memory usage)
#%%
df_hw.info()
df_hw_reset = df_hw.reset_index()
df_hw_reset.info()
df_no_hw_avg.info()
df_no_hw_avg_reset = df_no_hw_avg.reset_index()
df_no_hw_avg_reset.info()

#%% md
# 
# ##   Step 2.2: Merge df_hw with df_no_hw_avg_reset
#%%

merged_df = pd.merge(df_hw_reset, df_no_hw_avg_reset[['lat', 'lon', 'year', 'hour', 'UHI', 'UBWI']],
                     on=['lat', 'lon', 'year', 'hour'],
                     suffixes=('', '_avg'))
#%%
merged_df.info()
merged_df.head()
#%% md
# 
# ##  Step 2.3: Subtract the average UHI from the hourly UHI and store in a new column
#%%

merged_df['UHI_diff'] = merged_df['UHI'] - merged_df['UHI_avg']
merged_df['UBWI_diff'] = merged_df['UBWI'] - merged_df['UBWI_avg']
# Now, merged_df contains your original data along with the subtracted UHI values in 'UHI_diff'
merged_df  # To check the first few rows of the merged DataFrame
#%%
merged_df[['UHI_diff', 'UBWI_diff']].describe()
#%%
merged_df.info()
#%% md
# ## (Optional) Step 2.4: Validate the results by checking the UHI values for a specific location and time
#%%

row_index = 198
print(merged_df.iloc[row_index].UHI_avg)
a_row = merged_df.iloc[row_index]
df_no_hw_avg_reset.loc[(df_no_hw_avg_reset['lat'] == a_row.lat) & (df_no_hw_avg_reset['lon'] == a_row.lon) & (
                df_no_hw_avg_reset['year'] == a_row.year) & (
                df_no_hw_avg_reset['hour'] == a_row.hour)].UHI

#%% md
# #  3: Averaged data for each local hour 
#%% md
# ##  Step 3.1: Adjust to local hour
#%%
import pandas as pd
import numpy as np

def convert_time_to_local_and_add_hour(df):
    """
    Converts the UTC timestamp in the DataFrame to local time based on longitude and adds a column for the local hour.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns for latitude ('lat'), longitude ('lon'), and UTC timestamp ('time')
    
    Returns:
    pd.DataFrame: DataFrame with additional columns ('local_time' and 'local_hour') for the timestamp adjusted to local time and the hour extracted from the local time
    """
    # Function to calculate timezone offset from longitude
    def calculate_timezone_offset(longitude):
        return np.floor(longitude / 15.0).astype(int)  # Approximate, not accounting for DST or specific timezone rules

    # Calculate timezone offsets for each row based on longitude
    offsets = calculate_timezone_offset(df['lon'].values)

    # Adjust timestamps by the offsets
    df['local_time'] = df['time'] + pd.to_timedelta(offsets, unit='h')

    # Extract the hour from the 'local_time' and create a new column
    df['local_hour'] = df['local_time'].dt.hour

    return df

# # Assuming 'df' is your original DataFrame
# # Make sure 'time' column is in datetime format
# df['time'] = pd.to_datetime(df['time'])

# # Convert UTC times to local times based on longitude and add local hour
# df = convert_time_to_local_and_add_hour(df)

#%%
merged_df = convert_time_to_local_and_add_hour(merged_df)
merged_df

#%%
merged_df.info()
#%% md
# ##  Step 3.1.1 optionally save the merged_df to a parquet file
#%%
import pandas as pd

try:
    import pyarrow
    print(f"pyarrow version: {pyarrow.__version__}")
    print(f"pandas version: {pd.__version__}")

    # Check if pyarrow is used for data interchange
    if merged_df.to_parquet.__module__.startswith('pyarrow'):
        print("pandas is using pyarrow for data interchange")
    else:
        print("pandas is not using pyarrow for data interchange")

except ImportError:
    print("pyarrow is not installed")
#%%
# Save the merged DataFrame to a Parquet file
merged_df_path = os.path.join(summary_dir, 'local_hour_adjusted_variables.parquet')
merged_df.to_parquet(merged_df_path)

#%%
merged_feather_path = os.path.join(summary_dir, 'local_hour_adjusted_variables.feather')
merged_df.to_feather(merged_feather_path)
#%%
# test_df = pd.read_feather(merged_feather_path)
#%% md
# ##  Step 3.2 compute average based on local hour