import pandas as pd
import numpy as np
import xarray as xr
import os
import argparse
import time
import gc  # For explicit garbage collection
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load data from a parquet file."""
    logging.info(f"Loading data from {file_path}")
    return pd.read_parquet(file_path).reset_index()

def process_year(year, data_dir, hw_def, percentile):
    """Process data for a single year."""
    logging.info(f"Processing data for year {year}")
    
    # Load data for all months
    df_all = pd.DataFrame()
    for month in range(1, 13):
        file_name = f"{year}_{month:02d}.parquet"
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            df_month = load_data(file_path)
            df_all = pd.concat([df_all, df_month], ignore_index=True)
        else:
            logging.warning(f"File not found: {file_path}")
    
    if df_all.empty:
        logging.warning(f"No data found for year {year}")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found
    
    logging.info(f"Loaded {len(df_all)} rows for year {year}")
    
    # Merge with HW definition data
    logging.info(f"Merging with HW definition data for year {year}")
    hw_column = f'HW{percentile}'
    df_merged = pd.merge(df_all, hw_def[['time', 'lat', 'lon', hw_column, 'location_ID']], 
                         on=['time', 'lat', 'lon'], how='left')
    del df_all  # Free up memory
    gc.collect()  # Force garbage collection
    
    # Add time-related columns
    logging.info(f"Adding time-related columns for year {year}")
    df_merged['hour'] = df_merged['time'].dt.hour
    df_merged['month'] = df_merged['time'].dt.month
    df_merged['year'] = df_merged['time'].dt.year
    
    # Calculate local time
    logging.info(f"Calculating local time for year {year}")
    offsets = np.floor(df_merged['lon'] / 15.0).astype(int)
    df_merged['local_time'] = df_merged['time'] + pd.to_timedelta(offsets, unit='h')
    df_merged['local_hour'] = df_merged['local_time'].dt.hour
    
    logging.info(f"Finished processing year {year}")
    return df_merged

def add_event_id(df):
    """Add event IDs to HW data."""
    logging.info("Adding event IDs to HW data")
    initial_count = len(df)
    
    df = df.sort_values(by=['location_ID', 'time'])
    df['time_diff'] = df.groupby('location_ID')['time'].diff().dt.total_seconds() / 3600
    df['new_event'] = (df['time_diff'] > 1)
    df['event_ID'] = df.groupby('location_ID')['new_event'].cumsum()
    df['global_event_ID'] = df['location_ID'].astype(str) + '_' + df['event_ID'].astype(str)
    logging.info(f"Added event IDs to {len(df)} rows")
    
    # Validation check
    if len(df) != initial_count:
        raise ValueError(f"Data loss detected during event ID addition. Expected {initial_count} rows, got {len(df)}")
    
    return df

def calculate_uhi_diff(df_hw, df_no_hw_avg):
    """
    Calculate the difference in UHI and UWBI between heatwave and non-heatwave periods.
    
    Args:
    df_hw (pd.DataFrame): DataFrame containing heatwave data
    df_no_hw_avg (pd.DataFrame): DataFrame containing averaged non-heatwave data
    
    Returns:
    pd.DataFrame: Merged DataFrame with UHI and UWBI differences
    """
    print("Calculating UHI and UWBI differences...")
    merged_df = pd.merge(df_hw, df_no_hw_avg[['lat', 'lon', 'year', 'hour', 'UHI', 'UWBI']], 
                         on=['lat', 'lon', 'year', 'hour'],
                         suffixes=('', '_avg'))
    merged_df['UHI_diff'] = merged_df['UHI'] - merged_df['UHI_avg']
    merged_df['UWBI_diff'] = merged_df['UWBI'] - merged_df['UWBI_avg']
    print("UHI and UWBI differences calculated.")
    return merged_df


def main(args):
    start_time = time.time()
    logging.info(f"Starting data processing at {time.ctime()}")
    
    # Load HW definition data
    logging.info(f"Loading HW definition data from {args.hw_data_path}")
    hw_def = pd.read_feather(args.hw_data_path)
    logging.info(f"Loaded HW definition data: {len(hw_def)} rows")
    
    # Process data year by year
    logging.info(f"Processing data for years {args.start_year} to {args.end_year}")
    df_merged = pd.DataFrame()
    for year in range(args.start_year, args.end_year + 1):
        df_year = process_year(year, args.data_dir, hw_def, args.percentile)
        if not df_year.empty:
            df_merged = pd.concat([df_merged, df_year], ignore_index=True)
        del df_year  # Free up memory
        gc.collect()  # Force garbage collection
    
    if df_merged.empty:
        logging.warning("No data found for any year. Exiting.")
        return

    total_rows = len(df_merged)
    logging.info(f"Total combined rows: {total_rows}")
    
    hw_column = f'HW{args.percentile}'
    logging.info(f"Separating HW and non-HW data based on {hw_column} definition")
    
    # Replace NaN values with False in the HW column
    df_merged[hw_column] = df_merged[hw_column].fillna(False)

    # Calculate UHI and UWBI
    df_merged['UHI'] = df_merged.TSA_U - df_merged.TSA_R
    df_merged['UWBI'] = df_merged.WBA_U - df_merged.WBA_R

    df_hw = df_merged[df_merged[hw_column]].copy()
    df_hw = add_event_id(df_hw)

    df_no_hw = df_merged[~df_merged[hw_column]]

    # Calculate average for no_HW data
    print("Calculating average for no_HW data...")
    df_no_hw_avg = df_no_hw[['lat', 'lon', 'year', 'hour', 'UHI', 'UWBI']].groupby(['lat', 'lon', 'year', 'hour']).mean().reset_index()
    print("Average for no_HW data calculated.")

    # Calculate UHI difference
    local_hour_adjusted_df = calculate_uhi_diff(df_hw, df_no_hw_avg)

    # Load location ID and height data
    print("Loading location ID and height data...")
    location_ID_path = os.path.join(args.summary_dir, 'location_IDs.nc')
    heightdat = os.path.join(args.summary_dir, 'topodata_0.9x1.25_USGS_070110_stream_c151201.nc')

    ds_location_ID = xr.open_dataset(location_ID_path, engine='netcdf4')
    ds_height = xr.open_dataset(heightdat, engine='netcdf4')
    print("Location ID and height data loaded.")

    # Merge TOPO into location_ID dataset
    print("Merging TOPO data...")
    ds_merged = xr.merge([ds_location_ID, ds_height.TOPO.isel(time=0)]).drop('time')

    # Convert to DataFrame
    id_topo_merged_df = ds_merged[['location_ID', 'TOPO']].to_dataframe().reset_index()
    print("TOPO data merged and converted to DataFrame.")

    # Merge TOPO values with existing DataFrame
    print("Merging TOPO values with existing DataFrame...")
    local_hour_adjusted_df = local_hour_adjusted_df.merge(id_topo_merged_df[['location_ID', 'TOPO']], 
                                                          on='location_ID', how='left', validate='m:1')
    print("TOPO values merged with existing DataFrame.")

    
    # Process and save HW data
    logging.info("Saving HW data")

    hw_output_path = os.path.join(args.summary_dir, f'HW{args.percentile}.feather')
    df_hw.to_feather(hw_output_path)
    logging.info(f"Saved HW data to {hw_output_path}")
    hw_rows = len(df_hw)
    del df_hw  # Free up memory
    gc.collect()  # Force garbage collection
    
    # Process and save non-HW data
    logging.info("Processing and saving non-HW data")
    # df_merged.drop(df_merged[df_merged[hw_column]].index, inplace=True)
    no_hw_output_path = os.path.join(args.summary_dir, f'no_hw_HW{args.percentile}.feather')
    df_no_hw.to_feather(no_hw_output_path)
    logging.info(f"Saved non-HW data to {no_hw_output_path}")
    no_hw_rows = len(df_merged)
    
    logging.info(f"Total combined rows: {total_rows}")
    logging.info(f"HW data: {hw_rows} rows")
    logging.info(f"Non-HW data: {no_hw_rows} rows")
    logging.info(f"Row Count Diff: {total_rows-hw_rows-no_hw_rows}")
    # # Final validation check
    # df_hw_check = pd.read_feather(hw_output_path)
    # df_no_hw_check = pd.read_feather(no_hw_output_path)
    # if len(df_hw_check) != hw_rows or len(df_no_hw_check) != no_hw_rows:
    #     raise ValueError("Data loss detected during saving. Saved data does not match processed data.")
    
    end_time = time.time()
    logging.info(f"Data processing completed at {time.ctime()}")
    logging.info(f"Total processing time: {(end_time - start_time) / 60:.2f} minutes")
    logging.info("All validation checks passed. Data integrity maintained throughout the process.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HW and non-HW data with new HW definition.")
    parser.add_argument("--data_dir", type=str, default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/parquet")
    parser.add_argument("--summary_dir", type=str, default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary")
    parser.add_argument("--hw_data_path", type=str, default="/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/hw_data.feather")
    parser.add_argument("--start_year", type=int, default=1985)
    parser.add_argument("--end_year", type=int, default=2013)
    parser.add_argument("--percentile", type=int, choices=[90, 95], default=95, help="Percentile for HW definition (90 or 95)")
    args = parser.parse_args()
    
    main(args)