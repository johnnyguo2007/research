import xarray as xr
import pandas as pd
import os
import glob
from concurrent.futures import ProcessPoolExecutor

import cftime
import numpy as np
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


# Ensure the output directory exists
output_dir = '/tmpdata/summerized_data/i.e215.I2000Clm50SpGs.hw_production.02/monthly_avg_for_each_year'
os.makedirs(output_dir, exist_ok=True)

# Path for the log file
log_file_path = os.path.join(output_dir, 'processed_files.log')

def log_file_status(file_path, status):
    # Function to log the status of each file
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_path} - {status}\n')

def process_month(year, month):
    # Use glob to match files by year and month, then sort by day
    file_pattern = f'/media/jguo/external_data/simulation_output/archive/case/lnd/hist/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.{year}-{month:02d}-*-00000.nc'
    file_paths = sorted(glob.glob(file_pattern))

    if not file_paths:  # Log if no files are found for the month
        log_file_status(f'No files found for {year}-{month:02d}', "Missing")
        return None

    # Log the processed files
    for file_path in file_paths:
        log_file_status(file_path, "Processed")

    # Open the datasets and concatenate them along the time dimension
    ds = xr.open_mfdataset(file_paths, combine='nested', concat_dim='time')

    # Round your 'time' coordinate to the nearest hour
    ds['time'] = round_to_nearest_hour(ds['time'].values)

    # Group by hour of the day and compute the mean
    monthly_hourly_avg = ds.groupby('time.hour').mean('time')

    return monthly_hourly_avg

def process_year(year):
    # List to store monthly averages for the year
    monthly_averages = []

    for month in range(1, 13):
        monthly_avg = process_month(year, month)
        if monthly_avg is not None:
            monthly_averages.append(monthly_avg)

    # Concatenate the monthly averages into a single dataset for the year
    if monthly_averages:
        yearly_avg = xr.concat(monthly_averages, pd.Index(range(1, 13), name='month'))
        # Define the output file path for the year
        output_file = os.path.join(output_dir, f'yearly_avg_{year}.nc')
        yearly_avg.to_netcdf(output_file)
        print(f'Processed and saved yearly average for {year}')

# Main processing loop with parallelization
start_year = 1985
end_year = 2014

# # Sequential processing for each year
# for year in range(start_year, end_year + 1):
#     process_year(year)

# Using ProcessPoolExecutor to parallelize processing across years
with ProcessPoolExecutor(max_workers=36) as executor:
    # Submit a process for each year
    futures = [executor.submit(process_year, year) for year in range(start_year, end_year + 1)]

    # Wait for all futures to complete
    for future in futures:
        future.result()

