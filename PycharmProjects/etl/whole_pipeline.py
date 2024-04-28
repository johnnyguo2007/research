import os
import glob
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import yaml

import psutil
import time
from datetime import datetime

from uhi_etl import *


def main():
    """Main function that orchestrates the entire process."""
    # Load configuration from YAML
    config = load_config("config.yaml")

    show_memory_usage("Before run_extract_variables")
    # Extract variables from daily h1 files
    if config["run_extract_variables"]:
        extract_variables(config["daily_file_pattern"], config["daily_variables_list"],
                          config["daily_extracted_cols_file"], config["log_file_path"])

    show_memory_usage("Before run_hw_detection")
    # Determine urban grid and heatwave days
    if config["run_hw_detection"]:
        ds_one_monthly_data = xr.open_dataset(config["one_simu_result_monthly_file"])
        urban_non_null_mask = ds_one_monthly_data['TSA_U'].isel(time=0).notnull().drop('time')

        ds_hw = xr.open_dataset(config["daily_extracted_cols_file"])

        def detect_heatwave(tsa_r_np):
            tsa_r_np = np.atleast_1d(tsa_r_np)
            col_hw = np.full(tsa_r_np.shape, np.nan)

            for i in range(2, len(tsa_r_np)):
                if (tsa_r_np[i - 2] > kelvin_threshold and
                        tsa_r_np[i - 1] > kelvin_threshold and
                        tsa_r_np[i] > kelvin_threshold):
                    col_hw[i - 2:i + 1] = 1

            return col_hw

        hw = xr.apply_ufunc(
            detect_heatwave, ds_hw['TREFMXAV_R'],
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True,
            output_dtypes=[bool]
        )

        ds_hw['HW'] = hw
        ds_hw_filtered = ds_hw.where(urban_non_null_mask.broadcast_like(ds_hw), drop=False)
        ds_hw_filtered.to_netcdf(config["daily_grid_hw_file"])

        # Determine heatwave dates
        daily_hw_urban_count = hw.sum(dim=['lat', 'lon']).compute()
        hw_dates = daily_hw_urban_count.where(daily_hw_urban_count > 1, drop=True)
        hw_dates.to_netcdf(config["daily_hw_dates_file"])

    # Convert NetCDF to Zarr
    if config["run_convert_to_zarr"]:
        one_hourly_file = config["one_hourly_file"]
        ds_one_hourly_data = xr.open_dataset(one_hourly_file)
        vars = [var for var in ds_one_hourly_data.data_vars if (len(ds_one_hourly_data[var].dims) == 3)]

        ds_daily_grid_hw = xr.open_dataset(config["daily_grid_hw_file"])
        ds_daily_grid_hw = convert_time(ds_daily_grid_hw)

        for year in range(config["start_year"], config["end_year"] + 1):
            netcdf_to_zarr_process_year(config["sim_results_hourly_dir"], config["research_results_zarr_dir"],
                                        config["log_file_path"], year, vars, ds_daily_grid_hw)

    # Separate HW and No HW data
    if config["run_sep_hw_no_hw"]:
        ds = xr.open_zarr(os.path.join(config["research_results_zarr_dir"], '3Dvars'))
        ds['UHI'] = ds.TSA_U - ds.TSA_R
        ds['UWBI'] = ds.WBA_U - ds.WBA_R
        separate_hw_no_hw_process_in_chunks(ds=ds, chunk_size=24 * 3, zarr_path=config["research_results_zarr_dir"])

    show_memory_usage("Before run_zarr_to_parquet")
    # Convert Zarr to Parquet
    if config["run_zarr_to_parquet"]:
        zarr_to_dataframe(config["research_results_zarr_dir"], config["start_year"], config["end_year"],
                          config["research_results_parquet_dir"], 'HW', config["core_vars"])
        zarr_to_dataframe(config["research_results_zarr_dir"], config["start_year"], config["end_year"],
                          config["research_results_parquet_dir"], 'NO_HW', config["core_vars"])

    # aggregate merge compute difference and add local hour
    if config["run_prepare_for_ml"]:
        # Main script
        ml_col_list = config["parquet_col_list"]
        df_hw, df_no_hw = load_data_from_parquet(config["research_results_parquet_dir"],
                                                 config["start_year"], config["end_year"],
                                                 ml_col_list)

        # Prepare DataFrame by decomposing datetime
        df_hw = add_year_month_hour_cols(df_hw)
        df_no_hw = add_year_month_hour_cols(df_no_hw)
        show_memory_usage("After add_year_month_hour_cols")

        df_no_hw_avg = df_no_hw.groupby(['lat', 'lon', 'year', 'hour'])[['UHI', 'UWBI']].mean()
        show_memory_usage("After groupby")

        local_hour_adjusted_df: pd.DataFrame = calculate_uhi_diff(df_hw, df_no_hw_avg)
        show_memory_usage("After calculate_uhi_diff")

        # this is where the multi-index is removed and turned into a regular column
        local_hour_adjusted_df = convert_time_to_local_and_add_hour(local_hour_adjusted_df)
        show_memory_usage("After convert_time_to_local_and_add_hour")

        local_hour_adjusted_df.rename(columns=lambda x: x.replace('UBWI', 'UWBI'), inplace=True)

        ensure_directory_exists(config["subset_from_parquet_and_merged_file_path"])
        local_hour_adjusted_df.to_feather(config["subset_from_parquet_and_merged_file_path"])

    if config["run_add_location_id_event_id"]:
        # add location ID
        location_ds = xr.open_dataset(config["loc_id_path"])
        location_df = location_ds.to_dataframe().reset_index()

        # Merge the location_df with the local_hour_adjusted_df
        local_hour_adjusted_df = pd.merge(local_hour_adjusted_df, location_df, on=['lat', 'lon'], how='left')

        local_hour_adjusted_df = add_event_id(local_hour_adjusted_df)

        local_hour_adjusted_df.to_feather(config["var_with_id_path"])


if __name__ == "__main__":
    main()
