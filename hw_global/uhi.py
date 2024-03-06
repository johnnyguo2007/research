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

def spatial_mean(da):
    """Computes the spatial mean if lat and lon dimensions are present."""
    if 'lat' in da.dims and 'lon' in da.dims:
        return da.mean(dim=['lat', 'lon'])
    return da

# Convert cftime.DatetimeNoLeap to numpy.datetime64
def convert_time(ds):
    ds['time'] = [pd.Timestamp(time.strftime()) for time in ds['time'].values]
    return ds

#%% md
# # Study Urban PCT From Surface Data
#%%
fsurdat:str = "/home/jguo/projects/cesm/inputdata/lnd/clm2/surfdata_map/release-clm5.0.18/surfdata_0.9x1.25_hist_16pfts_Irrig_CMIP6_simyr2000_c190214.nc"
ds_sur = xr.open_mfdataset(fsurdat)

#%%
ds_sur

#%%

#%%


# Load the dataset
ds = xr.open_dataset(fsurdat)

# Access the PCT_URBAN variable
pct_urban = ds['PCT_URBAN']
#%%
ds
#%%
# https://bb.cgd.ucar.edu/cesm/threads/proportion-of-cities-in-surface-data.8046/
# PCT_URBAN is the percent of each urban density type. The density types in order are
# tall building district (TBD), high density (HD), and medium density (MD).
# If you change those percentages, e.g, increase them, then you'll need to decrease
# some other surface type (e.g., PCT_NATVEG, PCT_CROP, PCT_LAKE, etc.).
# The sum of PCT_URBAN, PCT_NATVEG, PCT_CROP, PCT_LAKE, PCT_GLACIER, PCT_WETLAND needs to be 100%.
# PCT_URBAN has multiple layers for different urban density types, sum across the 'numurbl' dimension to get total urban coverage
total_urban_pct = pct_urban.sum(dim='numurbl')

# Filter locations where total urban percentage is greater than 2%
# Using .where() to retain the data structure and metadata
masked_urban_areas = total_urban_pct.where(total_urban_pct > 2)
#%%
import pandas as pd
def find_top_urban_areas(masked_urban_areas):
    # Convert the stacked DataArray to a pandas DataFrame
    df = masked_urban_areas.stack(z=('lsmlat', 'lsmlon')).to_dataframe(name='urban_pct')

    # Use the nlargest method on the DataFrame to find the top 50 values
    top_urban_areas_df = df.nlargest(50, 'urban_pct')
    # Drop the redundant 'lsmlat' and 'lsmlon' columns
    top_urban_areas_df = top_urban_areas_df.drop(columns=['lsmlat', 'lsmlon'])

    return top_urban_areas_df

# Call the function with the masked_urban_areas variable
top_urban_areas = find_top_urban_areas(masked_urban_areas)
print(top_urban_areas)



#%%

# Plotting
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

# Get longitude and latitude information from the dataset
longitude = ds['LONGXY']
latitude = ds['LATIXY']

# Plotting the urban areas that meet the condition
# Note: masked_urban_areas already has values below 2% filtered out, so we use it directly
plt.pcolormesh(longitude, latitude, masked_urban_areas, transform=ccrs.PlateCarree(), cmap='cool') #cmap='cool') cmap='Reds')

plt.colorbar(label='Total Urban Percentage')
plt.title('Urban Areas with More Than 2% Urban Coverage')
plt.show()


#%%
masked_urban_areas.values.flatten()[~np.isnan(masked_urban_areas.values.flatten())][:10]
np.nanmax(masked_urban_areas.values), np.nanmin(masked_urban_areas.values)

#%% md
# # Use City Info from OpenStreetMap Overpass API to find the grid
#%%
# Load city data
cities_df = pd.read_csv('Data/cities_info.csv')

# Convert the 'Population' column to integers
cities_df['Population'] = pd.to_numeric(cities_df['Population'], errors='coerce')

# Filter cities with population greater than 50,000
cities_filtered = cities_df[cities_df['Population'] > 50000]
cities_filtered
#%% md
# ## join city data with surface data
# 
#%%


# Assuming the surface data is loaded into ds_sur and masked_urban_areas is already calculated

# Convert city coordinates to grid indices
# Assuming the grid starts at the minimum longitude and latitude in ds_sur
min_lon = ds_sur['LONGXY'].min()
min_lat = ds_sur['LATIXY'].min()

# Calculate grid indices for longitude and latitude
# The floor operation ensures the index aligns with the grid definition
cities_filtered.loc[:, 'lon_idx'] = np.floor((cities_filtered['Lon']) / 1.25).astype(int)
cities_filtered.loc[:, 'lat_idx'] = np.floor((cities_filtered['Lat']) / 0.9).astype(int)
# Initialize masked_urban_areas_from_city with the same shape and coordinates as masked_urban_areas, filled with np.nan
masked_urban_areas_from_city = xr.full_like(masked_urban_areas, np.nan)
# Update masked_urban_areas for each city
for lon_idx, lat_idx in zip(cities_filtered['lon_idx'], cities_filtered['lat_idx']):
    # Ensure indices are within bounds
    if 0 <= lon_idx < len(ds_sur['lsmlon']) and 0 <= lat_idx < len(ds_sur['lsmlat']):
        masked_urban_areas_from_city[lat_idx, lon_idx] = 1  # Or any other value indicating urban area

# Note: This approach uses minimal looping and relies more on vectorized operations for efficiency

#%%

masked_urban_areas_from_city
#%%
# Call the function with the masked_urban_areas variable
top_urban_areas_with_city_filter = find_top_urban_areas(masked_urban_areas_from_city)
print(top_urban_areas_with_city_filter)

#%% md
# # Data Retrieval
#     ## Select either Monethly (h0) daily (h1) or Hourly(h2) using wild card
#%%
# Directory where the netCDF files are located
# data_directory = '/home/jguo/projects/cesm/archive/case/lnd/hist/'  # Current directory. Adjust this if your files are elsewhere.
data_directory = '/tmpdata/i.e215.I2000Clm50SpGs.hw_production.02'

# File pattern
# h0 is 20 years of 24 monthly files, each file contains just one monthly output 
# h2 for hourly data i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.1986-12-31-00000.nc
#file_pattern = "i.e215.I2000Clm50SpGs.hw_spinup.03.clm2.h0.*.nc"
file_pattern = 'i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.*.nc'

file_path_pattern = os.path.join(data_directory, file_pattern)
#%% md
# ## Get the list of files using the pattern
#%%
file_list = glob.glob(file_path_pattern)
file_list.sort(key=lambda x: os.path.basename(x).split('.')[-2])

#%%
# os.listdir(directory)
len(file_list)
#%% md
# ## Open the files using Dask and Xarray
#%% md
# 
#%%
# List of variables to drop
# drop_vars = ["ZSOI", "DZSOI", "WATSAT", "SUCSAT", "BSW", "HKSAT", "ZLAKE", "DZLAKE"]

#%%
#ds = xr.open_mfdataset(file_list, combine='by_coords', engine='netcdf4', chunks={'time': 240})
ds = xr.open_mfdataset(file_list, combine='by_coords', engine='netcdf4')

#ds = xr.open_mfdataset(file_list, combine='by_coords', engine='netcdf4', drop_variables=drop_vars)

#%%

print(type(ds.dims['time']))

#%% md
# # Computing UHI
#  
# UHI calculation
#  We computed UHI2m (ΔT2m) and UHIs (ΔTs) from the variables generated by the urban and rural
#  sub-grids in the grid cells where our selected cities are located as below:
#  ΔT2m = 𝑇2m,urban  − 𝑇2, rural)()+. (1) (variable name: TSA_U, TSA_R)
#  ΔTs = 𝑇s,urban − 𝑇s, rural)()+. (2) 
#  where T2m,urban and T2m,rural are the urban and rural 2m-height air temperature respectively; Ts,urban
#  and Ts,rural are the urban and rural radiative surface temperature respectively. The 2m-height air
#  temperatures are obtained directly from the model default outputs; they are diagnostic rather than
#  prognostic model outputs since they are calculated from the surface fluxes. The radiative surface
#  temperatures (Ts) are derived from the emitting longwave radiation as:
#  = 1↑ 3 43 5 1↓
# 
#  (3)
#  where L↑ and L↓ are the incoming and outgoing longwave radiation fluxes, 𝜀 is the surface
#  emissivity (computed from land surface dataset in CLM 4), 𝜎 is the Stefan-Boltzmann constant
#  (5.67 × 10–8 W m–2 K–4). These longwave radiation fluxes were extracted from the urban and
#  vegetated subgrid land units in the grid cells where the selected cities are located. CLM
#  calculates the overall emitted longwave radiation from urban subgrid by weighting radiation
#  from different canyon facets according to their areal fractions, their sky-view factors, and the
#  canyon height to width ratio (Oleson et al. 2010a).
# 
#%%

#%% md
# ## examine the xarray object
#%%
ds
#%% md
# # Convert K to C for TBOT
#%%
ds['TBOT_C'] = ds['TBOT'] - 273.15

# Update the attributes for the new variable to reflect the change in units
ds['TBOT_C'].attrs['units'] = 'C'
ds['TBOT_C'].attrs['long_name'] = '2m air temperature in Celsius'
#%% md
# # Weighting and Averaging
# 
#%% md
# ## Area Weighted
# 
#%%
#pure land portion weight
#la = (ds.landfrac*ds.area).isel(time=0).drop(['time'])
#just area weight
la = (ds.area).isel(time=0).drop(['time'])
# the unit should not matter in the context of calculating weight
la = la * 1e6  #converts from land area from km2 to m2 
la.attrs['units'] = 'm^2'
lw = la
lw.plot() 
#%%
lw.sum().values
#%%
def compute_weighted_avg_vectorized(ds, lw):
    # Normalize the area weights to sum to 1
    lw_normalized = lw / lw.sum(dim=['lat', 'lon'])

    # Filter the dataset to include only data variables with 'lat' and 'lon' dimensions
    data_vars_with_latlon = {name: var for name, var in ds.data_vars.items() if {'lat', 'lon'}.issubset(var.dims)}

    # Use the filtered data variables to create a new dataset
    ds_with_latlon = xr.Dataset(data_vars_with_latlon)

    # Perform the vectorized operation of multiplying each variable by the area weights
    weighted_ds = ds_with_latlon * lw_normalized

    # Sum over the 'lat' and 'lon' dimensions to get the weighted average
    weighted_avg_ds = weighted_ds.sum(dim=['lat', 'lon'])

    return weighted_avg_ds

# Call the function with the dataset and land weights
weighted_avg_dataset = compute_weighted_avg_vectorized(ds, lw)

# weighted_avg_dataset now contains the weighted averages for each variable

#%% md
# ## yearly avg, the resample command is important and cool
#%%

yearly_avg = weighted_avg_dataset.resample(time='AS-FEB').mean()
#%%
yearly_avg
#print(yearly_avg.time[0])
#%% md
# # Define Variables to Report
#%%
variables = ['TBOT_C', 'QBOT', 'RAIN', 'EFLX_LH_TOT', 'FSH']
#%%
# Calculate spatial average for each variable
for var in variables:
    yearly_avg[var] = spatial_mean(yearly_avg[var])

#%% md
# # Convert time axis to integer year
#%%
yearly_avg = yearly_avg.assign_coords(time=yearly_avg.time.dt.year)
#%%
print(yearly_avg)
#%% md
# # Plot time series of yearly average
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")  # Use seaborn's whitegrid style
plt.rcParams['font.size'] = 12  # Increase default font size

# Assuming 'yearly_avg' is your data and 'variables' is a list of variable names you want to plot
# Create subplots for each variable
fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(16, 6 * len(variables)))

# Ensure 'axes' is an array even with a single subplot
if len(variables) == 1:
    axes = [axes]

colors = sns.color_palette("deep", len(variables))  # A palette of distinct colors for each variable

for ax, var, color in zip(axes, variables, colors):
    ax.plot(yearly_avg['time'], yearly_avg[var], label=var, color=color, linewidth=2)

    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel(f'{var} ({ds[var].attrs.get("units", "unknown units")})', fontsize=14)  # Retrieving units from dataset attributes

    # Fetching long_name from attributes and forming the title
    long_name = ds[var].attrs.get('long_name', var)  # Using var as default if long_name is absent
    ax.set_title(f'{long_name} ({var})', fontsize=16, fontweight='bold')

    ax.legend(loc='upper left')

    # Set x-axis ticks to show each year
    ax.set_xticks(yearly_avg['time'])  # Set ticks at each year in the 'time' array
    ax.set_xticklabels(yearly_avg['time'].values, rotation=45)  # Set tick labels to show the year, rotating for readability

plt.tight_layout(pad=3)  # Adjust padding
plt.show()

#%% md
# 