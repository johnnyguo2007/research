import numpy as np
import pandas as pd
import xarray as xr

# Read the Zarr dataset using Xarray with automatic chunking for Dask
# ds = xr.open_zarr('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/zarr2/HW', chunks='auto')
ds = xr.open_zarr('/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/zarr2/NO_HW', chunks='auto')
# Select core variables
# core_vars = ['UHI', 'UBWI', 'HW']
core_vars = ['UHI', 'UBWI']
ds = ds[core_vars]

# Define start and end year of your dataset
start_year = 1985
end_year = 2014

# Define the number of years to process in each chunk
years_per_chunk = 10

for start_chunk_year in range(start_year, end_year + 1, years_per_chunk):
    # Initialize an empty list to hold DataFrames for each year
    df_list = []

    # Determine the end year for the current chunk, ensuring it does not exceed the end_year
    end_chunk_year = min(start_chunk_year + years_per_chunk - 1, end_year)

    for year in range(start_chunk_year, end_chunk_year + 1):
        # Select the data for the current year
        ds_year = ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))

        # Convert to DataFrame without resetting the index
        df_year = ds_year.to_dataframe(['lat', 'lon', 'time']).dropna()

        # Append the DataFrame for the current year to the list
        df_list.append(df_year)

    # Concatenate all DataFrames in the list to create a single DataFrame for the 10-year chunk
    df_chunk = pd.concat(df_list)

    # Define the path to the Parquet file for this 10-year chunk
    parquet_file_path = f'/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/parquet/NO_HW_{start_chunk_year}_{end_chunk_year}.parquet'

    # Write the 10-year chunk DataFrame to a Parquet file
    df_chunk.to_parquet(parquet_file_path, engine='pyarrow', index=True)

    print(f'Saved {start_chunk_year}-{end_chunk_year} data to {parquet_file_path}')
