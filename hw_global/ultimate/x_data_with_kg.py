# Import necessary libraries
import pandas as pd
import os
import argparse
import sys
import mlflow

def filter_by_year(df, year):
    """Filter dataframe by a specific year."""
    return df[df['year'] == int(year)]

def filter_by_temperature_above_300(df, temperature):
    """Filter dataframe for temperatures above a specified threshold."""
    return df[df['temperature'] > float(temperature)]

def filter_by_KGMajorClass(df, major_class):
    """Filter dataframe by KGMajorClass."""
    return df[df['KGMajorClass'] == major_class]

def filter_by_hw_count(df, threshold):
    """
    Filter dataframe to include locations with hardware counts below or equal to the threshold.
    
    Parameters:
        df (DataFrame): The input dataframe.
        threshold (int): The maximum allowed hardware count.
    
    Returns:
        DataFrame: Filtered dataframe.
    """
    threshold = int(threshold)
    # Group by latitude, longitude, and year to count occurrences
    hw_counts = df[['lat', 'lon', 'year']].groupby(['lat', 'lon', 'year']).size().reset_index(name='count')
    # Identify locations to include based on the threshold
    locations_to_include = hw_counts[hw_counts['count'] <= threshold][['lat', 'lon']].drop_duplicates()
    # Merge to filter the original dataframe
    df = df.merge(locations_to_include, on=['lat', 'lon'], how='left', indicator=True)
    return df[df['_merge'] == 'left_only'].drop(columns=['_merge'])

def filter_by_uhi_diff_category(df, threshold, category):
    """
    Filter dataframe based on UHI difference category.
    
    Parameters:
        df (DataFrame): The input dataframe.
        threshold (float): The UHI difference threshold.
        category (str): Category to filter ('Positive', 'Insignificant', 'Negative').
    
    Returns:
        DataFrame: Filtered dataframe.
    
    Raises:
        ValueError: If an invalid category is provided.
    """
    if category == 'Positive':
        return df[df['UHI_diff'] > threshold]
    elif category == 'Insignificant':
        return df[(df['UHI_diff'] >= -threshold) & (df['UHI_diff'] <= threshold)]
    elif category == 'Negative':
        return df[df['UHI_diff'] < -threshold]
    else:
        raise ValueError("Invalid category. Choose 'Positive', 'Insignificant', or 'Negative'.")

def generate_x_data(summary_dir, merged_feather_file, time_period, filters, 
                   feature_column, delta_column, hw_nohw_diff_column, double_diff_column, delta_mode, daily_freq):
    """
    Generate X_data and X_data_with_kg datasets based on provided parameters.
    
    Parameters:
        summary_dir (str): Directory for saving summary files.
        merged_feather_file (str): Name of the merged feather file.
        time_period (str): Time period to filter ('day' or 'night').
        filters (str): Filters to apply, separated by semicolons.
        feature_column (str): Column name for features in schema.
        delta_column (str): Column name for delta variables in schema.
        hw_nohw_diff_column (str): Column name for hardware differences in schema.
        double_diff_column (str): Column name for double differencing in schema.
        delta_mode (str): Mode for delta variables ('none', 'include', 'only').
        daily_freq (bool): Whether to aggregate data to daily frequency.
    
    Returns:
        tuple: Two DataFrames, X and X_with_kg.
    """
    # Load data from Feather file
    merged_feather_path = os.path.join(summary_dir, merged_feather_file)
    print(f"Loading data from {merged_feather_path}")
    local_hour_adjusted_df = pd.read_feather(merged_feather_path)
    print(f"Loaded dataframe with shape: {local_hour_adjusted_df.shape}")

    # Apply specified filters
    if filters:
        print("Applying filters...")
        filter_function_pairs = filters.split(';')
        for filter_function_pair in filter_function_pairs:
            filter_parts = filter_function_pair.split(',')
            filter_function_name = filter_parts[0]
            filter_params = filter_parts[1:]
            if filter_function_name in globals():
                print(f"Applying filter: {filter_function_name} with parameters {filter_params}")
                local_hour_adjusted_df = globals()[filter_function_name](local_hour_adjusted_df, *filter_params)
            else:
                print(f"Warning: Filter function {filter_function_name} not found. Skipping.")
        print(f"Dataframe shape after applying filters: {local_hour_adjusted_df.shape}")

    # Load feature list from Excel schema
    df_daily_vars = pd.read_excel('/home/jguo/research/hw_global/Data/hourlyDataSchema.xlsx')

    # Select features based on specified columns
    daily_vars = df_daily_vars.loc[df_daily_vars[feature_column] == 'Y', 'Variable']
    daily_var_lst = daily_vars.tolist()

    delta_vars = df_daily_vars.loc[df_daily_vars[delta_column] == 'Y', 'Variable']
    hw_nohw_diff_vars = df_daily_vars.loc[df_daily_vars[hw_nohw_diff_column] == 'Y', 'Variable']
    double_diff_vars = df_daily_vars.loc[df_daily_vars[double_diff_column] == 'Y', 'Variable']

    # Append hardware and double differencing variables to the feature list
    daily_var_lst.extend([f"hw_nohw_diff_{var}" for var in hw_nohw_diff_vars])
    daily_var_lst.extend([f"Double_Differencing_{var}" for var in double_diff_vars])

    # Calculate delta variables if required
    if delta_mode in ["include", "only"]:
        for var in delta_vars:
            var_U = f"{var}_U"
            var_R = f"{var}_R"
            delta_var = f"delta_{var}"
            if var_U in local_hour_adjusted_df.columns and var_R in local_hour_adjusted_df.columns:
                local_hour_adjusted_df[delta_var] = local_hour_adjusted_df[var_U] - local_hour_adjusted_df[var_R]
            else:
                print(f"Warning: {var_U} or {var_R} not found in dataframe columns.")

        if delta_mode == "only":
            # Use only delta variables
            daily_var_lst = [f"delta_{var}" for var in delta_vars]
        else:
            # Include delta variables in the feature list
            daily_var_lst += [f"delta_{var}" for var in delta_vars]

    # Calculate double differencing variables
    for var in double_diff_vars:
        var_U = f"hw_nohw_diff_{var}_U"
        var_R = f"hw_nohw_diff_{var}_R"
        double_diff_var = f"Double_Differencing_{var}"
        if var_U in local_hour_adjusted_df.columns and var_R in local_hour_adjusted_df.columns:
            local_hour_adjusted_df[double_diff_var] = local_hour_adjusted_df[var_U] - local_hour_adjusted_df[var_R]
        else:
            print(f"Warning: {var_U} or {var_R} not found in dataframe columns.")

    # Define day and night masks based on local_hour
    daytime_mask = local_hour_adjusted_df['local_hour'].between(7, 16)
    nighttime_mask = (local_hour_adjusted_df['local_hour'].between(20, 24) | 
                      local_hour_adjusted_df['local_hour'].between(0, 6))

    # Extract date from local_time
    local_hour_adjusted_df['date'] = pd.to_datetime(local_hour_adjusted_df['local_time']).dt.date

    # Apply time period filter
    if time_period == "day":
        uhi_diff = local_hour_adjusted_df[daytime_mask]
    else:
        uhi_diff = local_hour_adjusted_df[nighttime_mask]

    # Aggregate to daily frequency if specified
    if daily_freq:
        uhi_diff = uhi_diff[daily_var_lst + ['UHI_diff', 'lat', 'lon', 'date']]
        uhi_diff = uhi_diff.groupby(['lat', 'lon', 'date']).mean().reset_index()

    # Prepare feature matrix X
    X = uhi_diff[daily_var_lst]

    # Add extra columns to create X_with_kg
    extra_cols = ['global_event_ID', 'lon', 'lat', 'local_hour', 'KGClass', 'KGMajorClass']
    X_with_kg = pd.concat([X, uhi_diff[extra_cols]], axis=1)

    return X, X_with_kg

if __name__ == "__main__":
    # Set up argument parser for command-line interface
    parser = argparse.ArgumentParser(description="Generate X_data feather files.")
    parser.add_argument("--summary_dir", type=str, required=True, help="Directory for saving summary files.")
    parser.add_argument("--merged_feather_file", type=str, required=True, help="Merged feather file name.")
    parser.add_argument("--time_period", choices=["day", "night"], required=True, help="Time period (day or night).")
    parser.add_argument("--filters", type=str, default="", help="Filters to apply to the dataframe.")
    parser.add_argument("--feature_column", type=str, default="X_vars2", help="Feature column name in df_daily_vars.")
    parser.add_argument("--delta_column", type=str, default="X_vars_delta", help="Delta column name in df_daily_vars.")
    parser.add_argument("--hw_nohw_diff_column", type=str, default="HW_NOHW_Diff", help="HW-NoHW diff column name.")
    parser.add_argument("--double_diff_column", type=str, default="Double_Diff", help="Double Differencing column name.")
    parser.add_argument("--delta_mode", choices=["none", "include", "only"], default="include", help="Delta mode.")
    parser.add_argument("--daily_freq", action="store_true", help="If set, calculate daily average.")
    parser.add_argument("--run_type", type=str, default="test", help="Beginning part of experiment name")
    parser.add_argument("--exp_name_extra", type=str, default="", help="Extra info that goes to the end of experiment name")

    args = parser.parse_args()

    # Set up MLflow experiment for logging
    experiment_name = f'{args.run_type}_{args.time_period.capitalize()}_{args.exp_name_extra}'
    mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()

    # Log command line arguments and the full command line
    for arg, value in vars(args).items():
        mlflow.log_param(arg, value)

    command_line = f"python {' '.join(sys.argv)}"
    mlflow.log_param("command_line", command_line)

    # Create directory for figures and artifacts
    figure_dir = os.path.join(args.summary_dir, 'mlflow', experiment_name)
    os.makedirs(figure_dir, exist_ok=True)

    # Generate the X_data and X_data_with_kg datasets
    X, X_with_kg = generate_x_data(
        args.summary_dir, 
        args.merged_feather_file, 
        args.time_period, 
        args.filters,
        args.feature_column, 
        args.delta_column, 
        args.hw_nohw_diff_column, 
        args.double_diff_column,
        args.delta_mode, 
        args.daily_freq
    )

    # Save the X_data dataframe to a Feather file
    X_path = os.path.join(figure_dir, 'X_data.feather')
    X.to_feather(X_path)
    print(f"Saved X data to {X_path}")

    # Save the X_data_with_kg dataframe to a Feather file
    X_with_kg_path = os.path.join(figure_dir, 'X_data_with_kg.feather')
    X_with_kg.to_feather(X_with_kg_path)
    print(f"Saved X data with KG to {X_with_kg_path}")

    # End the MLflow run
    mlflow.end_run()
    