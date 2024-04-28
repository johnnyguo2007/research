import xarray as xr
import numpy as np
import os

# Define global constants
LAT_LEN: int = 192
LON_LEN: int = 288

# Load the original dataset
one_simu_result_monthly_file = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/sim_results/monthly/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h0.1985-01.nc'
ds: xr.Dataset = xr.open_dataset(one_simu_result_monthly_file)


# Create a unique index for each lat-lon pair
location_ID: np.ndarray = np.arange(LAT_LEN * LON_LEN)  # This is a 1D array

# Create a new dataset with only lon, lat, and location_ID
location_ds: xr.Dataset = xr.Dataset({
    'lon': ds['lon'],
    'lat': ds['lat'],
    'location_ID': (('lat', 'lon'), location_ID.reshape(LAT_LEN, LON_LEN))
})

summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'
loc_id_path = os.path.join(summary_dir, 'location_IDs.nc')
# Save the new dataset with the location_ID
location_ds.to_netcdf(loc_id_path)
