# %%
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd

# %%


# Open the Zarr store
ds_no_hw_zarr = xr.open_zarr('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/zarr/NO_HW')
ds_hw_zarr = xr.open_zarr('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/zarr/HW')



# %%
ds_hw_zarr

# %%
def convert_time_to_local_and_add_hour(df):
    def calculate_timezone_offset(longitude):
        return np.floor(longitude / 15.0).astype(int)

    offsets = calculate_timezone_offset(df['lon'].values)
    df['local_time'] = df['time'] + pd.to_timedelta(offsets, unit='h')
    df['local_hour'] = df['local_time'].dt.hour
    return df

# %%
var = 'FSH'

# %%
data_array_no_hw = ds_no_hw_zarr[var]
mean_data_array_no_hw = data_array_no_hw.mean(dim='time')
data_array_hw = ds_hw_zarr[var]
mean_data_array_hw = data_array_hw.mean(dim='time')
diff_data_array = mean_data_array_hw - mean_data_array_no_hw

df_no_hw = mean_data_array_no_hw.to_dataframe().reset_index()
df_hw = mean_data_array_hw.to_dataframe().reset_index()
df_diff = diff_data_array.to_dataframe().reset_index()
df_diff 


# %%
df_diff.dropna(subset=['FSH'], inplace=True)

# %%


# %%
daytime_mask = local_hour_adjusted_df['local_hour'].between(8, 16)
nighttime_mask = (local_hour_adjusted_df['local_hour'].between(20, 24) | local_hour_adjusted_df['local_hour'].between(0, 4))

# %%
sandbox

# %%
df_diff 

# %%
#sort by FSH
df_diff.sort_values(by='FSH', inplace=True)
df_diff

# %%
# from ds_no_hw_zarr and ds_hw_zarr get the time series of FSH at a specific location (62.670158,30.00)
hw = ds_hw_zarr['FSH'].sel(lat=62.670158, lon=30.00)
no_hw = ds_no_hw_zarr['FSH'].sel(lat=62.670158, lon=30.00)

# %%
hw

# %%
no_hw

# %%
hw.mean().values

# %%

no_hw.mean().values

# %%
#filter out data after 2014
hw = hw.where(hw['time.year'] < 2014, drop=True)

# %%
no_hw = no_hw.where(no_hw['time.year'] < 2014, drop=True)

# %%
#hw has hourly data, now I only print hw unique dates
np.unique(hw['time'].dt.date.values).size


# %%
hw.plot()


# %%
df_hw = hw.to_dataframe().reset_index().dropna()

# %%
df_hw

# %%
df_no_hw = no_hw.to_dataframe().reset_index().dropna()
df_no_hw

# %%


# %%



