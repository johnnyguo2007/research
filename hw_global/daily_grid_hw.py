#%%
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import glob
import pandas as pd
import os
import cartopy.crs as ccrs
import numpy as np
import pandas as pd



#%% md
# ##  6.  Use TSA_U to determine urban grid
#%%
one_monthly_file = '/tmpdata/summerized_data/i.e215.I2000Clm50SpGs.hw_production.02/sub_sample/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h0.1985-01.nc'
ds_monthly = xr.open_dataset(one_monthly_file)

print(f"urban: {ds_monthly.TSA_U.notnull().sum().compute().item() *100/(288*192):.1f}%")

#%%
# Select the 'TSA_U' variable and get the non-null mask
# Using the .isel() method to select the first time slice
urban_non_null_mask = ds_monthly['TSA_U'].isel(time=0).notnull().drop('time')   

#%% md
# 
# #  Define HW Temporal filter
# ##  1.  Load the daily(h1 tape) simulation temperature data
#%% md
# Load the dataset, I copied the daily h1 files to /home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/daily_raw it is 137G in total
# I run the utils/extract_var_save.py to extract the subset of variables from the above directory
# and save the data to /home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/summary
#%% md
# 
#%%
hw_summary_dir = '/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/summary'
hw_output_file = 'i.e215.I2000Clm50SpGs.hw_production.02.clm2.h1.hwdaysOnly.nc'
hw_out_file_path = os.path.join(hw_summary_dir, hw_output_file)
#check if hw_file exists
hw_file_exist = os.path.isfile(hw_out_file_path)
#%%

#%% md
# 
# ##  2   For each grid cell the find time periods that satisfy the HW definition. 
#%% md
# We use the definition from the US National Weather Service (NWS): three or more consecutive days of maximumtemperature reaching at least 90 ◦F (32.2 ◦C). 
# We consider that, in each grid cell (a size on the order of 100 × 100 km), its rural sub-grid represents a local background environment for the city. 
# Therefore, for each city we use its rural 2m-height temperature (T2m,rural) to define HWs.
# 
# We use this variable in the daily h1 file: TREFMXAV_R:long_name = "Rural daily maximum of average 2-m temperature" ;


#if hw_file does not exist, then we need to run the following code to create the hw_file
if not hw_file_exist:
    # Open the NetCDF file containing the rural daily maximum of average 2-m temperature
    hw_input_file = 'i.e215.I2000Clm50SpGs.hw_production.02.clm2.h1.TSA_UR_TREFMXAV_R.nc'
    hw_input_file_path = os.path.join(hw_summary_dir, hw_input_file)
    ds_hw = xr.open_dataset(hw_input_file_path)
    ds_hw
    # Define the threshold temperature in Kelvin
    # Convert 90 degrees Fahrenheit to Kelvin
    fahrenheit_threshold = 90
    kelvin_threshold = (fahrenheit_threshold - 32) * (5/9) + 273.15  # 305.3722 K

    # Define a function to apply on each grid cell to detect heatwaves
    def detect_heatwave(tsa_r_np):
        # Ensure tsa_r_np is a 1D array for simplicity
        tsa_r_np = np.atleast_1d(tsa_r_np)
        hw = np.full(tsa_r_np.shape, np.nan)  # Initialize HW with NaN

        # Check for heatwaves
        for i in range(2, len(tsa_r_np)):
            if (tsa_r_np[i-2] > kelvin_threshold and
                    tsa_r_np[i-1] > kelvin_threshold and
                    tsa_r_np[i] > kelvin_threshold):
                hw[i-2:i+1] = 1  # Mark all three days as heatwave

        return hw

    # Use apply_ufunc to apply the detect_heatwave function across the dataset
    hw = xr.apply_ufunc(
        detect_heatwave, ds_hw['TREFMXAV_R'],
        input_core_dims=[['time']],  # Specify the core dimension
        output_core_dims=[['time']],  # Ensure output has the same core dimension as input
        vectorize=True,  # Enable broadcasting and looping over other dimensions
        output_dtypes=[float]  # Specify the output data type
    )
    # Optional: save the modified dataset to a new NetCDF file
    # Assign the HW data back to the original dataset as a new variable
    ds_hw['HW'] = hw
    ds_hw.to_netcdf(hw_out_file_path)   
else:
    # Load the existing HW data
    ds_hw =xr.open_dataset(hw_out_file_path)
    hw = ds_hw['HW']    

#%%
# Apply the mask to filter the dataset
# Apply the urban mask across all time points without dropping any
ds_hw_filtered = ds_hw.where(urban_non_null_mask.broadcast_like(ds_hw), drop=False)

#%%
# report 
#%% md
# 
#%%
hw = ds_hw_filtered['HW']


#%%
#save hw_dates to a file
hw.to_netcdf('/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/summary/daily_grid_hw.nc')
#%%
#
# Sum over the spatial dimensions to count the number of HW == 1 cells for each day
# Assuming 'lat' and 'lon' are the names of your spatial dimensions
daily_hw_urban_count = hw.sum(dim=['lat', 'lon']).compute()

#%%

#print out daily count of urban grid cells for days that has count > 1
hw_dates= daily_hw_urban_count.where(daily_hw_urban_count > 1, drop=True)
hw_dates.to_netcdf('/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/summary/daily_hw_dates.nc')
