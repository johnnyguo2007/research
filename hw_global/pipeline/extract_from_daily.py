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
    

def extract_variables(input_pattern, variables, output_file):
    print(f"Extracting variables {variables} from {input_pattern} into {output_file}")
    file_paths = sorted(glob.glob(input_pattern))

    # Log the processed files
    for file_path in file_paths:
        log_file_status(file_path, "Processed")

    # Open the datasets and concatenate them along the time dimension
    ds_var = xr.open_mfdataset(file_paths, combine='by_coords', data_vars=variables)

    # Convert to DataFrame Drop rows where TSA_U or TREFMXAV_R is null
    df = ds_var.to_dataframe().dropna(subset=['TSA_U', 'TREFMXAV_R']).reset_index()


    df['time'] = df['time'].apply(convert_cftime_to_datetime)

    # Save to Feather
    df.to_feather(output_file)

def main():
    file_pattern = f'/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/sim_results/daily/i.e215.I2000Clm50SpGs.hw_production.05.clm2.h1.*-00000.nc'
    variables_list = ['TSA', 'TSA_U', 'TSA_R', 'TREFMXAV_R']
    output_file = f'/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/i.e215.I2000Clm50SpGs.hw_production.05.clm2.h1.TSA_UR_TREFMXAV_R.feather'
    
    extract_variables(file_pattern, variables_list, output_file)

if __name__ == '__main__':
    main()