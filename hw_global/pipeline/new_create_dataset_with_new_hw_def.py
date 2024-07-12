import pandas as pd
import numpy as np
import xarray as xr
import os
import argparse
import time
import gc  # For explicit garbage collection

def load_data(file_path):
    """Load data from a parquet file."""
    print(f"Loading data from {file_path}")
    return pd.read_parquet(file_path).reset_index()

def process_year(year, data_dir, hw_def, percentile):
    """Process data for a single year."""
    print(f"Processing data for year {year}")
    hw_file = os.path.join(data_dir, f"ALL_HW_{year}.parquet")
    no_hw_file = os.path.join(data_dir, f"ALL_NO_HW_{year}.parquet")
    
    # Load data
    df_hw = load_data(hw_file)
    df_no_hw = load_data(no_hw_file)
    print(f"Loaded {len(df_hw)} HW rows and {len(df_no_hw)} non-HW rows for year {year}")
    
    # Combine data
    initial_count = len(df_hw) + len(df_no_hw)
    df_all = pd.concat([df_hw, df_no_hw], ignore_index=True)
    del df_hw, df_no_hw  # Free up memory
    gc.collect()  # Force garbage collection
    print(f"Combined data for year {year}: {len(df_all)} total rows")
    
    # Validation check
    if len(df_all) != initial_count:
        raise ValueError(f"Data loss detected in year {year}. Expected {initial_count} rows, got {len(df_all)}")
    
    # Merge with HW definition data
    print(f"Merging with HW definition data for year {year}")
    hw_column = f'HW{percentile}'
    df_merged = pd.merge(df_all, hw_def[['time', 'lat', 'lon', hw_column, 'location_ID']], 
                         on=['time', 'lat', 'lon'], how='left')
    del df_all  # Free up memory
    gc.collect()  # Force garbage collection
    
    # Validation check
    if len(df_merged) != initial_count:
        raise ValueError(f"Data loss detected during merge in year {year}. Expected {initial_count} rows, got {len(df_merged)}")
    
    # Add time-related columns
    print(f"Adding time-related columns for year {year}")
    df_merged['hour'] = df_merged['time'].dt.hour
    df_merged['month'] = df_merged['time'].dt.month
    df_merged['year'] = df_merged['time'].dt.year
    
    # Calculate local time
    print(f"Calculating local time for year {year}")
    offsets = np.floor(df_merged['lon'] / 15.0).astype(int)
    df_merged['local_time'] = df_merged['time'] + pd.to_timedelta(offsets, unit='h')
    df_merged['local_hour'] = df_merged['local_time'].dt.hour
    
    print(f"Finished processing year {year}")
    return df_merged

def add_event_id(df):
    """Add event IDs to HW data."""
    print("Adding event IDs to HW data")
    initial_count = len(df)
    
    df = df.sort_values(by=['location_ID', 'time'])
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
    
    # Load HW definition data
    print(f"Loading HW definition data from {args.hw_data_path}")
    hw_def = pd.read_feather(args.hw_data_path)
    print(f"Loaded HW definition data: {len(hw_def)} rows")
    
    # Process data year by year
    print(f"Processing data for years {args.start_year} to {args.end_year}")
    df_merged = pd.DataFrame()
    for year in range(args.start_year, args.end_year + 1):
        df_year = process_year(year, args.data_dir, hw_def, args.percentile)
        df_merged = pd.concat([df_merged, df_year], ignore_index=True)
        del df_year  # Free up memory
        gc.collect()  # Force garbage collection
    
    print(f"Total combined rows: {len(df_merged)}")
    
    hw_column = f'HW{args.percentile}'
    print(f"Separating HW and non-HW data based on {hw_column} definition")
    
    # Replace NaN values with False in the HW column
    df_merged[hw_column] = df_merged[hw_column].fillna(False)
    
    # Process and save HW data
    print("Processing and saving HW data")
    df_hw = df_merged[df_merged[hw_column]].copy()
    df_hw = add_event_id(df_hw)
    hw_output_path = os.path.join(args.summary_dir, f'HW{args.percentile}.feather')
    df_hw.to_feather(hw_output_path)
    print(f"Saved HW data to {hw_output_path}")
    hw_rows = len(df_hw)
    del df_hw  # Free up memory
    gc.collect()  # Force garbage collection
    
    # Process and save non-HW data
    print("Processing and saving non-HW data")
    df_merged.drop(df_merged[df_merged[hw_column]].index, inplace=True)
    no_hw_output_path = os.path.join(args.summary_dir, f'no_hw_HW{args.percentile}.feather')
    df_merged.to_feather(no_hw_output_path)
    print(f"Saved non-HW data to {no_hw_output_path}")
    no_hw_rows = len(df_merged)
    
    print(f"HW data: {hw_rows} rows")
    print(f"Non-HW data: {no_hw_rows} rows")
    
    # Final validation check
    df_hw_check = pd.read_feather(hw_output_path)
    df_no_hw_check = pd.read_feather(no_hw_output_path)
    if len(df_hw_check) != hw_rows or len(df_no_hw_check) != no_hw_rows:
        raise ValueError("Data loss detected during saving. Saved data does not match processed data.")
    
    end_time = time.time()
    print(f"Data processing completed at {time.ctime()}")
    print(f"Total processing time: {(end_time - start_time) / 60:.2f} minutes")
    print("All validation checks passed. Data integrity maintained throughout the process.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HW and non-HW data with new HW definition.")
    parser.add_argument("--data_dir", type=str, default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/parquet")
    parser.add_argument("--summary_dir", type=str, default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/hw95_summary")
    parser.add_argument("--hw_data_path", type=str, default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/hw95_summary/hw_data.feather")
    parser.add_argument("--start_year", type=int, default=1985)
    parser.add_argument("--end_year", type=int, default=2013)
    parser.add_argument("--percentile", type=int, choices=[90, 95], default=95, help="Percentile for HW definition (90 or 95)")
    args = parser.parse_args()
    
    main(args)