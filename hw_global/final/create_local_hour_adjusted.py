import pandas as pd
import numpy as np
import os

def convert_time_to_local_and_add_hour(df):
    def calculate_timezone_offset(longitude):
        return np.floor(longitude / 15.0).astype(int)

    offsets = calculate_timezone_offset(df['lon'].values)
    df['local_time'] = df['time'] + pd.to_timedelta(offsets, unit='h')
    df['local_hour'] = df['local_time'].dt.hour
    return df

def load_and_process_data(data_dir, start_year, end_year):
    df_hw_list = []
    df_no_hw_list = []

    for year in range(start_year, end_year + 1):
        file_name_hw = os.path.join(data_dir, f"ALL_HW_{year}_{year}.parquet")
        file_name_no_hw = os.path.join(data_dir, f"ALL_NO_HW_{year}_{year}.parquet")

        df_hw_tmp = pd.read_parquet(file_name_hw)
        df_no_hw_tmp = pd.read_parquet(file_name_no_hw)

        df_hw_list.append(df_hw_tmp)
        df_no_hw_list.append(df_no_hw_tmp)

    df_hw = pd.concat(df_hw_list)
    df_no_hw = pd.concat(df_no_hw_list)

    df_hw.index = df_hw.index.set_levels([df_hw.index.levels[0], df_hw.index.levels[1], pd.to_datetime(df_hw.index.levels[2])])
    df_hw['hour'] = df_hw.index.get_level_values('time').hour
    df_hw['month'] = df_hw.index.get_level_values('time').month
    df_hw['year'] = df_hw.index.get_level_values('time').year

    df_no_hw.index = df_no_hw.index.set_levels([df_no_hw.index.levels[0], df_no_hw.index.levels[1], pd.to_datetime(df_no_hw.index.levels[2])])
    df_no_hw['hour'] = df_no_hw.index.get_level_values('time').hour
    df_no_hw['year'] = df_no_hw.index.get_level_values('time').year
    df_no_hw_avg = df_no_hw.groupby(['lat', 'lon', 'year', 'hour']).mean()

    return df_hw, df_no_hw_avg

def calculate_uhi_diff(df_hw, df_no_hw_avg):
    df_hw_reset = df_hw.reset_index()
    df_no_hw_avg_reset = df_no_hw_avg.reset_index()

    merged_df = pd.merge(df_hw_reset, df_no_hw_avg_reset[['lat', 'lon', 'year', 'hour', 'UHI', 'UBWI']],
                         on=['lat', 'lon', 'year', 'hour'],
                         suffixes=('', '_avg'))

    merged_df['UHI_diff'] = merged_df['UHI'] - merged_df['UHI_avg']
    merged_df['UBWI_diff'] = merged_df['UBWI'] - merged_df['UBWI_avg']

    return merged_df

def main():
    data_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/parquet'
    summary_dir = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.02/research_results/summary'
    start_year = 1985
    end_year = 2013

    df_hw, df_no_hw_avg = load_and_process_data(data_dir, start_year, end_year)
    merged_df = calculate_uhi_diff(df_hw, df_no_hw_avg)
    merged_df = convert_time_to_local_and_add_hour(merged_df)

    merged_df_path = os.path.join(summary_dir, 'local_hour_adjusted_variables.parquet')
    merged_df.to_parquet(merged_df_path)

    merged_feather_path = os.path.join(summary_dir, 'local_hour_adjusted_variables.feather')
    merged_df.to_feather(merged_feather_path)

if __name__ == "__main__":
    main()