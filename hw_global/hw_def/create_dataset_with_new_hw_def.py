import pandas as pd
import numpy as np
import xarray as xr
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def load_data(file_path):
    print(f"Loading data from {file_path}")
    return pd.read_parquet(file_path)

def process_year(year, data_dir, hw_def):
    print(f"Processing data for year {year}")
    hw_file = os.path.join(data_dir, f"ALL_HW_{year}.parquet")
    no_hw_file = os.path.join(data_dir, f"ALL_NO_HW_{year}.parquet")
    
    df_hw = load_data(hw_file)
    df_no_hw = load_data(no_hw_file)
    print(f"Loaded {len(df_hw)} HW rows and {len(df_no_hw)} non-HW rows for year {year}")
    
    initial_count = len(df_hw) + len(df_no_hw)
    df_all = pd.concat([df_hw, df_no_hw])
    df_all.reset_index(inplace=True)
    print(f"Combined data for year {year}: {len(df_all)} total rows")
    
    # Validation check
    if len(df_all) != initial_count:
        raise ValueError(f"Data loss detected in year {year}. Expected {initial_count} rows, got {len(df_all)}")
    
    print(f"Merging with HW definition data for year {year}")
    df_merged = pd.merge(df_all, hw_def[['time', 'lat', 'lon', 'HW95', 'location_ID']], 
                         on=['time', 'lat', 'lon'], how='left')
    
    # Validation check
    if len(df_merged) != len(df_all):
        raise ValueError(f"Data loss detected during merge in year {year}. Expected {len(df_all)} rows, got {len(df_merged)}")
    
    print(f"Adding time-related columns for year {year}")
    df_merged['hour'] = df_merged['time'].dt.hour
    df_merged['month'] = df_merged['time'].dt.month
    df_merged['year'] = df_merged['time'].dt.year
    
    print(f"Calculating local time for year {year}")
    offsets = np.floor(df_merged['lon'] / 15.0).astype(int)
    df_merged['local_time'] = df_merged['time'] + pd.to_timedelta(offsets, unit='h')
    df_merged['local_hour'] = df_merged['local_time'].dt.hour
    
    print(f"Finished processing year {year}")
    return df_merged

def add_event_id(df):
    print("Adding event IDs to HW data")
    initial_count = len(df)
    df.sort_values(by=['location_ID', 'time'], inplace=True)
    df['time_diff'] = df.groupby('location_ID')['time'].diff().dt.total_seconds() / 3600
    df['new_event'] = (df['time_diff'] > 1)
    df['event_ID'] = df.groupby('location_ID')['new_event'].cumsum()
    df['global_event_ID'] = df['location_ID'].astype(str) + '_' + df['event_ID'].astype(str)
    print(f"Added event IDs to {len(df)} rows")
    
    # Validation check
    if len(df) != initial_count:
        raise ValueError(f"Data loss detected during event ID addition. Expected {initial_count} rows, got {len(df)}")
    
    return df

def main(args):
    start_time = time.time()
    print(f"Starting data processing at {time.ctime()}")
    print(f"Loading HW definition data from {args.hw_data_path}")
    hw_def = pd.read_feather(args.hw_data_path)
    print(f"Loaded HW definition data: {len(hw_def)} rows")
    
    print(f"Processing data for years {args.start_year} to {args.end_year}")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_year, year, args.data_dir, hw_def) 
                   for year in range(args.start_year, args.end_year + 1)]
        
        df_list = []
        for future in as_completed(futures):
            df_list.append(future.result())
    
    print("Combining data from all years")
    df_merged = pd.concat(df_list)
    print(f"Total combined rows: {len(df_merged)}")
    
    # Validation check
    expected_total = sum(len(df) for df in df_list)
    if len(df_merged) != expected_total:
        raise ValueError(f"Data loss detected during combination. Expected {expected_total} rows, got {len(df_merged)}")
    
    print("Separating HW and non-HW data based on HW95 definition")
    df_hw = df_merged[df_merged['HW95']]
    df_no_hw = df_merged[~df_merged['HW95']]
    print(f"HW data: {len(df_hw)} rows")
    print(f"Non-HW data: {len(df_no_hw)} rows")
    
    # Validation check
    if len(df_hw) + len(df_no_hw) != len(df_merged):
        raise ValueError(f"Data loss detected during separation. Expected {len(df_merged)} total rows, got {len(df_hw) + len(df_no_hw)}")
    
    df_hw = add_event_id(df_hw)
    
    hw_output_path = os.path.join(args.summary_dir, 'HW95.feather')
    no_hw_output_path = os.path.join(args.summary_dir, 'no_hw_HW95.feather')
    
    print(f"Saving HW data to {hw_output_path}")
    df_hw.to_feather(hw_output_path)
    print(f"Saving non-HW data to {no_hw_output_path}")
    df_no_hw.to_feather(no_hw_output_path)
    
    # Final validation check
    df_hw_check = pd.read_feather(hw_output_path)
    df_no_hw_check = pd.read_feather(no_hw_output_path)
    if len(df_hw_check) != len(df_hw) or len(df_no_hw_check) != len(df_no_hw):
        raise ValueError("Data loss detected during saving. Saved data does not match processed data.")
    
    end_time = time.time()
    print(f"Data processing completed at {time.ctime()}")
    print(f"Total processing time: {(end_time - start_time) / 60:.2f} minutes")
    print("All validation checks passed. Data integrity maintained throughout the process.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HW and non-HW data with new HW definition.")
    parser.add_argument("--data_dir", type=str, default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/parquet")
    parser.add_argument("--summary_dir", type=str, default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/hw95_summary")
    parser.add_argument("--hw_data_path", type=str, default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/hw95_summary/hw_data.feather")
    parser.add_argument("--start_year", type=int, default=1985)
    parser.add_argument("--end_year", type=int, default=2013)
    args = parser.parse_args()
    
    main(args)
    