import pandas as pd
import os

# Define directory and filenames
input_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'
input_filename = 'no_hw_HW98.feather'
output_filename = 'no_HW98_group_by_location_ID_year_local_hour.feather'

# Construct full file paths
input_path = os.path.join(input_dir, input_filename)
output_path = os.path.join(input_dir, output_filename)

# Read the feather file
df = pd.read_feather(input_path)

# List of columns to remove
columns_to_remove = ['time', 'HW98', 'event_ID', 'global_event_ID', 'month', 'local_time']

# Remove specified columns
df = df.drop(columns=columns_to_remove)

# Group by location_ID, year, local_hour
grouped = df.groupby(['location_ID', 'year', 'local_hour'])

# Compute average for all other columns, except lat and lon
agg_dict = {col: 'mean' for col in df.columns if col not in ['location_ID', 'year', 'local_hour', 'lat', 'lon']}
agg_dict['lat'] = 'first'
agg_dict['lon'] = 'first'

result = grouped.agg(agg_dict)

# Reset index to make grouped columns into regular columns
result = result.reset_index()

# Save the result to a new feather file
result.to_feather(output_path)

print(f"Processing complete. New feather file saved at: {output_path}")