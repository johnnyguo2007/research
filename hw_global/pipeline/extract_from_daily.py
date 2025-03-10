import xarray as xr
import pandas as pd
import os
import glob
import numpy as np

# Ensure the output directory is current dir
output_dir = os.path.dirname(os.path.realpath(__file__))

# Path for the log file
log_file_path = os.path.join(output_dir, 'processed_files.log')

def log_file_status(file_path, status):
    # Function to log the status of each file
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_path} - {status}\n')

def convert_cftime_to_datetime(ct):
    return pd.Timestamp(ct.year, ct.month, ct.day)

def extract_variables_for_year(input_pattern, variables, year, output_file):
    print(f"Extracting variables {variables} for year {year} from {input_pattern}")
    file_paths = sorted(glob.glob(input_pattern.format(year=year)))

    # Log the processed files
    for file_path in file_paths:
        log_file_status(file_path, "Processed")

    # Process files in smaller chunks
    chunk_size = 2  # Adjust this value based on your memory constraints
    all_data = []
    for i in range(0, len(file_paths), chunk_size):
        chunk_files = file_paths[i:i+chunk_size]
        
        # Open the datasets and concatenate them along the time dimension
        ds_var = xr.open_mfdataset(chunk_files, combine='by_coords')
        ds_var = ds_var[variables]
        # Convert to DataFrame Drop rows where TSA_U or TREFMXAV_R is null
        df = ds_var.to_dataframe().dropna(subset=['TSA_U', 'TREFMXAV_R']).reset_index()

        df['time'] = df['time'].apply(convert_cftime_to_datetime)

        all_data.append(df)

        # Clear memory
        del ds_var, df

    # Concatenate all data for the year
    year_data = pd.concat(all_data, ignore_index=True)
    return year_data

def main():
    base_pattern = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/sim_results/daily/i.e215.I2000Clm50SpGs.hw_production.05.clm2.h1.{year}*-00000.nc'
    variables_list = ['TSA', 'TSA_U', 'TSA_R', 'TREFMXAV_R']
    output_file = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/i.e215.I2000Clm50SpGs.hw_production.05.clm2.h1.TSA_UR_TREFMXAV_R.feather'
    
    all_years_data = []
    
    # Assuming the simulation runs from 1985 to 2013
    for year in range(1985, 2014):
        input_pattern = base_pattern.format(year=year)
        year_data = extract_variables_for_year(input_pattern, variables_list, year, output_file)
        all_years_data.append(year_data)
        print(f"Processed year {year}")

    # Concatenate all years' data
    final_data = pd.concat(all_years_data, ignore_index=True)

    # Write to feather file once
    final_data.to_feather(output_file)

    print(f"All data saved to {output_file}")

if __name__ == '__main__':
    main()