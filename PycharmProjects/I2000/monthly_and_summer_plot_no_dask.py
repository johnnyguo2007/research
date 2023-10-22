import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import pandas as pd
import cftime
import os
import seaborn as sns

def spatial_mean(da):
    """Computes the spatial mean if lat and lon dimensions are present."""
    if 'lat' in da.dims and 'lon' in da.dims:
        return da.mean(dim=['lat', 'lon'])
    return da

# Convert cftime.DatetimeNoLeap to numpy.datetime64
def convert_time(ds):
    ds['time'] = [pd.Timestamp(time.strftime()) for time in ds['time'].values]
    return ds

# Directory where the netCDF files are located
data_directory = '/home/jguo/projects/cesm/archive/case/lnd/hist/'

# File pattern
file_pattern = "i.e21.I2000Clm50SpGs.f09_g17.keerZ2year.003.clm2.h2.*.nc"
file_path_pattern = os.path.join(data_directory, file_pattern)

# Get the list of files using the pattern
file_list = glob.glob(file_path_pattern)
file_list.sort(key=lambda x: os.path.basename(x).split('.')[-2])

# Open the files using Xarray
ds = xr.open_mfdataset(file_list, combine='by_coords', engine='netcdf4')

# Convert the cftime.DatetimeNoLeap objects to pandas Timestamps
ds = convert_time(ds)

# Compute monthly average
monthly_avg = ds.resample(time='1M').mean()

# Calculate spatial average for each variable
monthly_avg['TBOT'] = spatial_mean(monthly_avg['TBOT'])
monthly_avg['QBOT'] = spatial_mean(monthly_avg['QBOT'])
monthly_avg['RAIN'] = spatial_mean(monthly_avg['RAIN'])

# Styling
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# Create subplots for each variable
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 18))

variables = ['TBOT', 'QBOT', 'RAIN']
colors = sns.color_palette("deep", 3)

for ax, var, color in zip(axes, variables, colors):
    ax.plot(monthly_avg['time'], monthly_avg[var], label=var, color=color, linewidth=2)

    ax.set_xlabel('Year-Month', fontsize=14)
    ax.set_ylabel(f'{var} ({ds[var].attrs.get("units", "unknown units")})', fontsize=14)

    long_name = ds[var].attrs.get('long_name', var)
    ax.set_title(f'{long_name} ({var})', fontsize=16, fontweight='bold')

    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout(pad=3)
plt.show()
