import pandas as pd
import xarray as xr
import numpy as np

# Load city data from CSV
cities_df = pd.read_csv('Data/cities_info.csv')

# Load the surface data file
fsurdat:str = "/home/jguo/projects/cesm/inputdata/lnd/clm2/surfdata_map/release-clm5.0.18/surfdata_0.9x1.25_hist_16pfts_Irrig_CMIP6_simyr2000_c190214.nc"
ds= xr.open_mfdataset(fsurdat)




# Access the 'PCT_URBAN' variable
pct_urban = ds['PCT_URBAN']

# Compute the actual values if it's a Dask array
pct_urban = pct_urban.compute()

# Calculate the difference between the max and min values across the 'numurbl' dimension
diff = pct_urban.max(dim='numurbl') - pct_urban.min(dim='numurbl')

# Flatten the array of differences to make it easier to sort and find top values
flat_diff = diff.values.flatten()

# Get the indices of the top 10 differences
top_10_indices = np.argpartition(flat_diff, -10)[-10:]

# Convert these 1D indices back to 2D grid coordinates
top_10_coords = np.unravel_index(top_10_indices, diff.shape)

# Print out the top 10 differences, their corresponding grid coordinates, and the values for all 3 slices
for i, index in enumerate(zip(*top_10_coords)):
    lat_index, lon_index = index
    # Access the PCT_URBAN values for all urban density types at the specific grid point
    urban_values = pct_urban.isel(lsmlat=lat_index, lsmlon=lon_index).values
    print(f"Rank {i+1}: Grid Point ({lat_index}, {lon_index}) - Difference: {flat_diff[top_10_indices[i]]}")
    for density_type, value in enumerate(urban_values):
        print(f"  Density Type {density_type}: {value}")

#
# # Access the required variables from the netCDF file
# longitude = ds['LONGXY'].values
# latitude = ds['LATIXY'].values
# pct_urban = ds['PCT_URBAN'].sum(dim='numurbl')  # Assuming PCT_URBAN needs to be summed across urban types
#
# # Prepare longitude and latitude grids for broadcasting
# lon_grid, lat_grid = np.meshgrid(longitude[0, :], latitude[:, 0])  # Adjust if longitude and latitude are 1D
#
# # Reshape city coordinates for broadcasting
# city_lons = cities_df['Lon'].values[:, np.newaxis, np.newaxis]
# city_lats = cities_df['Lat'].values[:, np.newaxis, np.newaxis]
#
# # Compute absolute differences using broadcasting
# lon_diff = np.abs(lon_grid - city_lons)
# lat_diff = np.abs(lat_grid - city_lats)
#
# # Find the indices of the grid cell closest to each city
# lon_idx = np.argmin(lon_diff, axis=2)
# lat_idx = np.argmin(lat_diff, axis=1)
#
# # Initialize list to store matched data
# matched_data = []
#
# # Iterate over the cities using the index
# for city_index in range(len(cities_df)):
#     lon_i = lon_idx[city_index]  # Longitude index for the city
#     lat_i = lat_idx[city_index]  # Latitude index for the city
#
#     # Extract the urban value and ensure it's loaded as a NumPy array
#     urban_value = pct_urban.isel(lsmlat=lat_i, lsmlon=lon_i).values
#
#     # If you need a scalar value and you're sure it's a single value, you can use .item()
#     if urban_value.size == 1:
#         urban_value = urban_value.item()
#
#     # Append matched data
#     matched_data.append({
#         'LONGXY': longitude[0, lon_i],
#         'LATIXY': latitude[lat_i, 0],
#         'PCT_URBAN': urban_value,
#         'Name': cities_df.iloc[city_index]['Name'],
#         'Lat': cities_df.iloc[city_index]['Lat'],
#         'Lon': cities_df.iloc[city_index]['Lon'],
#         'lsmlon': lon_i,
#         'lsmlat': lat_i
#     })
#
#
# # Convert matched data to DataFrame
# matched_df = pd.DataFrame(matched_data)
#
# # Display the DataFrame
# matched_df