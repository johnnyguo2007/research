import xarray as xr
import pandas as pd
import os
import glob

import cftime
import numpy as np




# Ensure the output directory is current dir
output_dir = os.path.dirname(os.path.realpath(__file__))

# Path for the log file
log_file_path = output_dir + '/processed_files.log'

def log_file_status(file_path, status):
    # Function to log the status of each file
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_path} - {status}\n')





import argparse

def extract_variables(input_pattern, variables, output_file):
    # Your code to extract variables from the input files and save to output file
    print(f"Extracting variables {variables} from {input_pattern} into {output_file}")
    # Use glob to match files by year and month, then sort by day
    file_paths = sorted(glob.glob(input_pattern))


    # Log the processed files
    for file_path in file_paths:
        log_file_status(file_path, "Processed")

    # Open the datasets and concatenate them along the time dimension
    ds = xr.open_mfdataset(file_paths, combine='by_coords')
    ds_var = ds[variables]
    ds_var.to_netcdf(output_file)

def main():
    # Extract variables from the DAILY h1 so we can detect heatwaves
    #file_pattern = f'/media/jguo/external_data/simulation_output/archive/case/lnd/hist/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h2.{year}-{month:02d}-*-00000.nc'
    file_pattern = f'/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/sim_results/hourly/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h1.*-00000.nc'
    variables_list = ['TSA', 'TSA_U', 'TSA_R', 'TREFMXAV_R']
    output_file = f'/home/jguo/process_data/i.e215.I2000Clm50SpGs.hw_production.02/summary/i.e215.I2000Clm50SpGs.hw_production.02.clm2.h1.TSA_UR_TREFMXAV_R.nc'
    # Call the function to extract variables using the provided arguments
    #extract_variables(args.input_pattern, variables_list, args.output_file)
    extract_variables(file_pattern, variables_list, output_file)


if __name__ == '__main__':
    main()

