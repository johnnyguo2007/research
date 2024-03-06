import xarray as xr
import pandas as pd
import os
import glob

import xarray as xr
import numpy as np
import cftime


#  Function to round cftime.DatetimeNoLeap objects to the nearest hour
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






# Use glob to match files by year and month, then sort by day
file_pattern = f'/tmpdata/i.e215.I2000Clm50SpGs.hw_production.02/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.1985-11-*-00000.nc'
file_paths = sorted(glob.glob(file_pattern))

# Open the datasets and concatenate them along the time dimension
ds = xr.open_mfdataset(file_paths, combine='nested', concat_dim='time')

# Assuming 'ds' is your Dataset and 'time' is your time coordinate
# Round your 'time' coordinate to the nearest hour
ds['time'] = round_to_nearest_hour(ds['time'].values)

# Group by hour of the day and compute the mean
monthly_hourly_avg = ds.groupby('time.hour').mean('time')

# Print the result
print(monthly_hourly_avg["hour"])
