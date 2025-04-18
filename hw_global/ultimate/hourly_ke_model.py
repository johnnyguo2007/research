#!/usr/bin/env python

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from catboost import CatBoostRegressor, Pool
import shap
import mlflow
import mlflow.catboost

from scipy.stats import linregress

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_by_year(df, year):
    """Filters the dataframe by year."""
    logging.info(f"Filtering data by year: {year}")
    return df[df['year'] == int(year)]

def filter_by_temperature_above_300(df, temperature):
    """Filters the dataframe by temperature above a given threshold."""
    logging.info(f"Filtering data with temperature above {temperature}")
    return df[df['temperature'] > float(temperature)]

def filter_by_KGMajorClass(df, major_class):
    """Filters the dataframe by KGMajorClass."""
    logging.info(f"Filtering data by KGMajorClass: {major_class}")
    return df[df['KGMajorClass'] == major_class]

def filter_by_hw_count(df, threshold):
    """Filters the dataframe by the number of heatwave events per location."""
    logging.info(f"Filtering data by HW count threshold: {threshold}")
    threshold = int(threshold)
    hw_counts = df[['lat', 'lon', 'year']].groupby(['lat', 'lon', 'year']).size().reset_index(name='count')
    locations_to_include = hw_counts[hw_counts['count'] <= threshold][['lat', 'lon']].drop_duplicates()
    df = df.merge(locations_to_include, on=['lat', 'lon'], how='left', indicator=True)
    return df[df['_merge'] == 'left_only'].drop(columns=['_merge'])

def filter_by_uhi_diff_category(df, threshold, category):
    """Filters the dataframe by UHI_diff category (Positive, Insignificant, or Negative)."""
    logging.info(f"Filtering data by UHI_diff category: {category}, threshold: {threshold}")
    threshold = float(threshold)
    if category == 'Positive':
        return df[df['UHI_diff'] > threshold]
    elif category == 'Insignificant':
        return df[(df['UHI_diff'] >= -threshold) & (df['UHI_diff'] <= threshold)]
    elif category == 'Negative':
        return df[df['UHI_diff'] < -threshold]
    else:
        raise ValueError("Invalid category. Choose 'Positive', 'Insignificant', or 'Negative'.")

def clear_gpu_memory():
    """Clears GPU memory if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared GPU memory.")

def get_long_name(var_name, df_daily_vars):
    """
    Retrieve the long name for a given variable name from the dataframe.
    If the variable name starts with 'delta_', it indicates a difference and the function
    will return a modified long name indicating the difference.
    """
    if df_daily_vars is None:
        logging.warning("df_daily_vars is None. Returning original var_name.")
        return var_name
    
    if var_name.startswith('delta_'):
        original_var_name = var_name.replace('delta_', '')
        original_long_name = df_daily_vars.loc[df_daily_vars['Variable'] == original_var_name, 'Long Name'].values
        if original_long_name.size > 0:
            return f"Difference of {original_long_name[0]}"
        else:
            return f"Difference of {original_var_name} (No long name found)"
    elif var_name.startswith('hw_nohw_diff_'):
        original_var_name = var_name.replace('hw_nohw_diff_', '')
        original_long_name = df_daily_vars.loc[df_daily_vars['Variable'] == original_var_name, 'Long Name'].values
        if original_long_name.size > 0:
            return f"HW Non-HW Difference of {original_long_name[0]}"
        else:
            return f"HW Non-HW Difference of {original_var_name} (No long name found)"
    elif var_name.startswith('Double_Differencing_'):
        original_var_name = var_name.replace('Double_Differencing_', '')
        original_long_name = df_daily_vars.loc[df_daily_vars['Variable'] == original_var_name, 'Long Name'].values
        if original_long_name.size > 0:
            return f"Double Difference of {original_long_name[0]}"
        else:
            return f"Double Difference of {original_var_name} (No long name found)"
    else:
        long_name = df_daily_vars.loc[df_daily_vars['Variable'] == var_name, 'Long Name'].values
        if long_name.size > 0:
            return long_name[0]
        else:
            return f"{var_name} (No long name found)"

def add_long_name(input_df, join_column='Feature', df_daily_vars=None):
    """
    Add a 'Long Name' column to the input dataframe by mapping the feature names
    to their corresponding long names using the get_long_name function.
    """
    input_df['Long Name'] = input_df[join_column].apply(lambda x: get_long_name(x, df_daily_vars))
    return input_df

def get_feature_group(feature_name):
    """
    Extracts the feature group from the feature name.
    """
    prefixes = ['delta_', 'hw_nohw_diff_', 'Double_Differencing_']
    for prefix in prefixes:
        if feature_name.startswith(prefix):
            return feature_name.replace(prefix, '')
    return feature_name

def save_and_log_artifact(filename, directory):
    plt.gcf().set_size_inches(15, 10)  # Set consistent figure size
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path)
    plt.close()

def get_shap_feature_importance(shap_values, feature_names, df_daily_vars):
    """
    Calculate the SHAP feature importance and return a dataframe with feature names,
    their importance, and percentage contribution. Also, add long names to the features.
    """
    shap_feature_importance = np.abs(shap_values).mean(axis=0)
    total_importance = np.sum(shap_feature_importance)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_feature_importance,
        'Percentage': (shap_feature_importance / total_importance) * 100
    })
    shap_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    shap_importance_df = add_long_name(shap_importance_df, join_column='Feature', df_daily_vars=df_daily_vars)
    return shap_importance_df

def process_shap_values(shap_values, feature_names, X, shap_df_additional_columns, time_period, df_daily_vars, output_dir, exclude_features=None):
    """
    Process SHAP values to calculate feature importance, generate plots, and save artifacts.
    """
    # Create directories for saving plots and artifacts
    post_process_shap_dir = os.path.join(output_dir, 'post_process_shap')
    feature_group_shap_dir = os.path.join(output_dir, 'feature_group_shap')
    os.makedirs(post_process_shap_dir, exist_ok=True)
    os.makedirs(feature_group_shap_dir, exist_ok=True)
    
    # Calculate SHAP feature importance
    shap_feature_importance = get_shap_feature_importance(shap_values, feature_names, df_daily_vars)
    
    # Exclude specified features from the SHAP summary plot
    if exclude_features:
        shap_feature_importance = shap_feature_importance[~shap_feature_importance['Feature'].isin(exclude_features)]
        logging.info(f"Excluding features from SHAP summary plot: {exclude_features}")
    
    # Generate SHAP summary plot for individual features
    logging.info(f"Creating SHAP summary plot for individual features for {time_period}time...")
    shap.summary_plot(
        shap_values[:, shap_feature_importance.index],
        X.iloc[:, shap_feature_importance.index],
        feature_names=shap_feature_importance['Feature'].tolist(),
        show=False
    )
    summary_output_path = f"post_process_{time_period}_shap_summary_plot.png"
    save_and_log_artifact(summary_output_path, post_process_shap_dir)
    mlflow.log_artifact(os.path.join(post_process_shap_dir, summary_output_path))
    plt.clf()
    
    # Generate SHAP value-based importance plot
    logging.info(f"Creating SHAP value-based importance plot for {time_period}time...")
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_feature_importance['Importance'].values,
            base_values=0,  # Base value is not meaningful in this context
            feature_names=shap_feature_importance['Feature'].tolist()
        ),
        show=False
    )
    plt.title(f'SHAP Value-based Importance Plot for {time_period.capitalize()}time')
    importance_output_path = f"post_process_{time_period}_shap_importance_plot.png"
    save_and_log_artifact(importance_output_path, post_process_shap_dir)
    mlflow.log_artifact(os.path.join(post_process_shap_dir, importance_output_path))
    plt.clf()
    
    # Calculate SHAP feature importance by group
    shap_feature_importance['Feature Group'] = shap_feature_importance['Feature'].apply(lambda x: get_feature_group(x))
    shap_feature_importance_by_group = shap_feature_importance.groupby('Feature Group')['Importance'].sum().reset_index()
    total_importance = shap_feature_importance_by_group['Importance'].sum()
    shap_feature_importance_by_group['Percentage'] = (shap_feature_importance_by_group['Importance'] / total_importance) * 100
    shap_feature_importance_by_group.sort_values(by='Importance', ascending=False, inplace=True)
    
    # Normalize feature values
    logging.info("Normalizing feature values...")
    X_normalized = X.copy()
    for col in X.columns:
        max_val = X[col].max()
        min_val = X[col].min()
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1e-8  # Avoid division by zero
        X_normalized[col] = (X[col] - min_val) / range_val
    logging.info("Feature values normalized using min-max scaling.")

    # Calculate group-level SHAP values and normalized feature values
    group_shap_values = []
    group_feature_values = []
    feature_names_arr = np.array(feature_names)
    for group in shap_feature_importance_by_group['Feature Group']:
        # Get features in this group
        group_features = shap_feature_importance['Feature'][shap_feature_importance['Feature Group'] == group].tolist()
        group_indices = [list(feature_names).index(feat) for feat in group_features]
        
        # Sum SHAP values for all features in the group
        group_shap = shap_values[:, group_indices].sum(axis=1)
        group_shap_values.append(group_shap)
        
        # Sum normalized feature values for all features in the group
        group_feature_value = X_normalized.iloc[:, group_indices].sum(axis=1).values
        group_feature_values.append(group_feature_value)

    group_shap_values = np.array(group_shap_values).T  # Transform to shape (n_samples, n_groups)
    group_feature_values = np.array(group_feature_values).T  # Transform to shape (n_samples, n_groups)
    
    # Save SHAP feature importance by group
    group_csv_path = f'{time_period}_shap_feature_importance_by_group.csv'
    shap_feature_importance_by_group.to_csv(os.path.join(feature_group_shap_dir, group_csv_path), index=False)
    mlflow.log_artifact(os.path.join(feature_group_shap_dir, group_csv_path))
    logging.info(f"Saved SHAP feature importance by group data to {group_csv_path}")
    
    # Create and log SHAP waterfall plot by feature group
    logging.info(f"Creating SHAP waterfall plot by feature group for {time_period}time...")
    plt.figure(figsize=(12, 0.5 * len(shap_feature_importance_by_group)))
    
    # Convert importance values to percentages
    waterfall_values = shap_feature_importance_by_group['Percentage'].values
    base_value = 0  # Start from 0% since we're showing percentages
    
    # Create waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            values=waterfall_values,
            base_values=base_value,
            data=None,
            feature_names=shap_feature_importance_by_group['Feature Group'].tolist()
        ),
        show=False,
        max_display=len(shap_feature_importance_by_group)
    )
    
    plt.xlabel('Percentage Contribution')
    plt.title(f'SHAP Waterfall Plot by Feature Group for {time_period.capitalize()}time')
    waterfall_output_path = f"post_process_{time_period}_shap_waterfall_plot_by_group.png"
    save_and_log_artifact(waterfall_output_path, feature_group_shap_dir)
    mlflow.log_artifact(os.path.join(feature_group_shap_dir, waterfall_output_path))
    plt.clf()
    
    # Create and log SHAP summary plot by feature group
    logging.info(f"Creating SHAP summary plot by feature group for {time_period}time...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        group_shap_values,
        group_feature_values,
        feature_names=shap_feature_importance_by_group['Feature Group'].tolist(),
        show=False
    )
    plt.title(f'SHAP Summary Plot by Feature Group for {time_period.capitalize()}time')
    summary_group_output_path = f"post_process_{time_period}_shap_summary_plot_by_group.png"
    save_and_log_artifact(summary_group_output_path, feature_group_shap_dir)
    mlflow.log_artifact(os.path.join(feature_group_shap_dir, summary_group_output_path))
    plt.clf()
    
    # Create and save percentage contribution plot
    logging.info(f"Creating percentage contribution plot for {time_period}time...")
    plt.figure(figsize=(12, 0.5 * len(shap_feature_importance)))
    plt.barh(shap_feature_importance['Long Name'], shap_feature_importance['Percentage'], 
             align='center', color='#ff0051')
    plt.title(f'Feature Importance (%) for {time_period.capitalize()}time')
    plt.xlabel('Percentage Contribution')
    plt.gca().invert_yaxis()
    
    for i, percentage in enumerate(shap_feature_importance['Percentage']):
        plt.text(percentage, i, f' {percentage:.1f}%', va='center')

    plt.tight_layout()
    percentage_output_path = f"post_process_{time_period}_percentage_contribution_plot.png"
    save_and_log_artifact(percentage_output_path, post_process_shap_dir)
    mlflow.log_artifact(os.path.join(post_process_shap_dir, percentage_output_path))
    plt.clf()

    # Save SHAP feature importance data to CSV
    shap_importance_path = os.path.join(post_process_shap_dir, f'{time_period}_shap_feature_importance.csv')
    shap_feature_importance.to_csv(shap_importance_path, index=False)
    mlflow.log_artifact(shap_importance_path)
    logging.info(f"Saved SHAP feature importance data to {shap_importance_path}")

    # Return the shap_feature_importance and shap_feature_importance_by_group if needed
    return shap_feature_importance, shap_feature_importance_by_group

def process_shap_values_by_kg_major_class(shap_values, feature_names, X, shap_df_additional_columns, time_period, df_daily_vars, output_dir, exclude_features=None):
    """
    Process SHAP values by KGMajorClass to calculate feature importance, generate plots, and save artifacts.
    """
    # Extract KGMajorClass for partitioning
    if 'KGMajorClass' in shap_df_additional_columns.columns:
        kg_major_classes = shap_df_additional_columns['KGMajorClass'].unique()
    else:
        logging.error("KGMajorClass column not found in shap_values DataFrame.")
        return

    kg_major_dir = os.path.join(output_dir, 'KGMajor')
    os.makedirs(kg_major_dir, exist_ok=True)

    for kg_major_class in kg_major_classes:
        class_indices = shap_df_additional_columns.index[shap_df_additional_columns['KGMajorClass'] == kg_major_class].tolist()
        if not class_indices:
            continue

        class_shap_values = shap_values[class_indices]
        class_X = X.iloc[class_indices]

        # Calculate SHAP feature importance for the class
        shap_feature_importance_class = get_shap_feature_importance(class_shap_values, feature_names, df_daily_vars)

        # Exclude specified features from the SHAP summary plot
        if exclude_features:
            shap_feature_importance_class = shap_feature_importance_class[~shap_feature_importance_class['Feature'].isin(exclude_features)]
            logging.info(f"Excluding features from SHAP summary plot for KGMajorClass '{kg_major_class}': {exclude_features}")

        # Create directories for the class
        class_dir = os.path.join(kg_major_dir, kg_major_class)
        os.makedirs(class_dir, exist_ok=True)

        # Generate SHAP summary plot for individual features
        logging.info(f"Creating SHAP summary plot for individual features for {time_period}time and KGMajorClass '{kg_major_class}'...")
        shap.summary_plot(
            class_shap_values[:, shap_feature_importance_class.index],
            class_X.iloc[:, shap_feature_importance_class.index],
            feature_names=shap_feature_importance_class['Feature'].tolist(),
            show=False
        )
        summary_output_path = f"{kg_major_class}_{time_period}_shap_summary_plot.png"
        save_and_log_artifact(summary_output_path, class_dir)
        plt.clf()

        # Generate SHAP value-based importance plot for the class
        logging.info(f"Creating SHAP value-based importance plot for {time_period}time and KGMajorClass '{kg_major_class}'...")
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_feature_importance_class['Importance'].values,
                base_values=0,  # Base value is not meaningful here
                feature_names=shap_feature_importance_class['Feature'].tolist()
            ),
            show=False
        )
        plt.title(f'SHAP Value-based Importance Plot for {kg_major_class} ({time_period.capitalize()}time)')
        importance_output_path = f"{kg_major_class}_{time_period}_shap_importance_plot.png"
        save_and_log_artifact(importance_output_path, class_dir)
        plt.clf()

        # Calculate SHAP feature importance by group for the class
        shap_feature_importance_class['Feature Group'] = shap_feature_importance_class['Feature'].apply(lambda x: get_feature_group(x))
        shap_feature_importance_by_group_class = shap_feature_importance_class.groupby('Feature Group')['Importance'].sum().reset_index()
        total_importance_class = shap_feature_importance_by_group_class['Importance'].sum()
        shap_feature_importance_by_group_class['Percentage'] = (shap_feature_importance_by_group_class['Importance'] / total_importance_class) * 100
        shap_feature_importance_by_group_class.sort_values(by='Importance', ascending=False, inplace=True)

        # Normalize feature values for the class
        class_X_normalized = class_X.copy()
        for col in class_X.columns:
            max_val = class_X[col].max()
            min_val = class_X[col].min()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1e-8  # Avoid division by zero
            class_X_normalized[col] = (class_X[col] - min_val) / range_val
        logging.info(f"Feature values normalized for KGMajorClass '{kg_major_class}' using min-max scaling.")

        # Calculate group-level SHAP values and normalized feature values for the class
        group_shap_values_class = []
        group_feature_values_class = []
        for group in shap_feature_importance_by_group_class['Feature Group']:
            # Get features in this group
            group_features = shap_feature_importance_class['Feature'][shap_feature_importance_class['Feature Group'] == group].tolist()
            group_indices = [feature_names.index(feat) for feat in group_features]
            
            # Sum SHAP values for all features in the group
            group_shap_class = class_shap_values[:, group_indices].sum(axis=1)
            group_shap_values_class.append(group_shap_class)
            
            # Sum normalized feature values for all features in the group
            group_feature_value_class = class_X_normalized.iloc[:, group_indices].sum(axis=1).values
            group_feature_values_class.append(group_feature_value_class)

        group_shap_values_class = np.array(group_shap_values_class).T  # Transform to shape (n_samples, n_groups)
        group_feature_values_class = np.array(group_feature_values_class).T  # Transform to shape (n_samples, n_groups)

        # Save SHAP feature importance by group for the class
        group_csv_path_class = f"{kg_major_class}_{time_period}_shap_feature_importance_by_group.csv"
        shap_feature_importance_by_group_class.to_csv(os.path.join(class_dir, group_csv_path_class), index=False)

        # Create and log SHAP waterfall plot by feature group for the class
        logging.info(f"Creating SHAP waterfall plot by feature group for {kg_major_class} ({time_period}time)...")
        plt.figure(figsize=(12, 0.5 * len(shap_feature_importance_by_group_class)))
        
        # Convert importance values to percentages
        waterfall_values_class = shap_feature_importance_by_group_class['Percentage'].values
        base_value_class = 0  # Start from 0% since we're showing percentages
        
        # Create waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=waterfall_values_class,
                base_values=base_value_class,
                data=None,
                feature_names=shap_feature_importance_by_group_class['Feature Group'].tolist()
            ),
            show=False,
            max_display=len(shap_feature_importance_by_group_class)
        )
        
        plt.xlabel('Percentage Contribution')
        plt.title(f'SHAP Waterfall Plot by Feature Group for {kg_major_class} ({time_period.capitalize()}time)')
        waterfall_output_path_class = f"{kg_major_class}_{time_period}_shap_waterfall_plot_by_group.png"
        save_and_log_artifact(waterfall_output_path_class, class_dir)
        plt.clf()

        # Create and log SHAP summary plot by feature group for the class
        logging.info(f"Creating SHAP summary plot by feature group for {kg_major_class} ({time_period}time)...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            group_shap_values_class,
            group_feature_values_class,
            feature_names=shap_feature_importance_by_group_class['Feature Group'].tolist(),
            show=False
        )
        plt.title(f'SHAP Summary Plot by Feature Group for {kg_major_class} ({time_period.capitalize()}time)')
        summary_group_output_path_class = f"{kg_major_class}_{time_period}_shap_summary_plot_by_group.png"
        save_and_log_artifact(summary_group_output_path_class, class_dir)

        # Create and save percentage contribution plot for the class
        logging.info(f"Creating percentage contribution plot for {kg_major_class} ({time_period}time)...")
        plt.figure(figsize=(12, 0.5 * len(shap_feature_importance_class)))
        plt.barh(shap_feature_importance_class['Long Name'], shap_feature_importance_class['Percentage'], 
                 align='center', color='#ff0051')
        plt.title(f'Feature Importance (%) for {kg_major_class} ({time_period.capitalize()}time)')
        plt.xlabel('Percentage Contribution')
        plt.gca().invert_yaxis()
        
        for i, percentage in enumerate(shap_feature_importance_class['Percentage']):
            plt.text(percentage, i, f' {percentage:.1f}%', va='center')

        plt.tight_layout()
        percentage_output_path_class = f"{kg_major_class}_{time_period}_percentage_contribution_plot.png"
        save_and_log_artifact(percentage_output_path_class, class_dir)

        # Save SHAP feature importance data to CSV for the class
        shap_importance_path_class = os.path.join(class_dir, f"{kg_major_class}_{time_period}_shap_feature_importance.csv")
        shap_feature_importance_class.to_csv(shap_importance_path_class, index=False)
        logging.info(f"SHAP feature importance data for {kg_major_class} saved to {shap_importance_path_class}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run UHI model for day, night, or hourly data.")
    parser.add_argument("--time_period", choices=["day", "night", "hourly"], required=True,
                        help="Specify whether to run for day, night, or hourly data.")
    parser.add_argument("--summary_dir", type=str, required=True, help="Directory for saving summary files and artifacts.")
    parser.add_argument("--merged_feather_file", type=str, required=True,
                        help="File name of the merged feather file containing the dataset.")
    parser.add_argument("--iterations", type=int, default=100000, help="Number of iterations for the CatBoost model.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the CatBoost model.")
    parser.add_argument("--depth", type=int, default=10, help="Depth of the trees for the CatBoost model.")
    parser.add_argument("--filters", type=str, default="",
                        help="Comma-separated list of filter function names and parameters to apply to the dataframe." +
                             "Multiple filters can be chained using semicolons. " +
                             "Example: --filters filter_by_year,2020;filter_by_temperature_above_300,305")
    parser.add_argument("--run_type", type=str, default="test", help="Beginning part of experiment name")
    parser.add_argument("--exp_name_extra", type=str, default="", help="Extra info that goes to the end of experiment name")
    parser.add_argument("--shap_calculation", action="store_true",
                        help="If set, SHAP-related calculations and graphs will be performed.")
    parser.add_argument("--feature_column", type=str, default="X_vars2",
                        help="Column name in df_daily_vars to select features")
    parser.add_argument("--delta_column", type=str, default="X_vars_delta",
                        help="Column name in df_daily_vars to select delta features")
    parser.add_argument("--hw_nohw_diff_column", type=str, default="HW_NOHW_Diff",
                        help="Column name in df_daily_vars to select HW-NoHW diff features")
    parser.add_argument("--double_diff_column", type=str, default="Double_Diff",
                        help="Column name in df_daily_vars to select Double Differencing features")
    parser.add_argument("--delta_mode", choices=["none", "include", "only"], default="include",
                        help="'none': don't use delta variables, 'include': use both original and delta variables, 'only': use only delta variables")
    parser.add_argument("--feature_selection", action="store_true", help="If set, perform feature selection using RFECV")
    parser.add_argument("--num_features", type=int, default=None,
                        help="Number of features to select using CatBoost's feature selection. If provided, --feature_selection will be ignored.")
    parser.add_argument("--daily_freq", action="store_true",
                        help="If set, data will be averaged by day before training the model.")
    parser.add_argument('--exclude_features', nargs='*', default=[], help='List of features to exclude from SHAP summary plot')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    logging.info("Starting UHI model script...")
    
    # Set up MLflow experiment
    summary_dir = args.summary_dir
    experiment_name = f'{args.run_type}_{args.time_period.capitalize()}_{args.exp_name_extra}'
    
    logging.info(f"Setting up MLflow experiment: {experiment_name}")
    mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")  # Set your MLflow tracking URI
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    
    # Log command line arguments and the full command line
    for arg, value in vars(args).items():
        mlflow.log_param(arg, value)
    
    command_line = f"python {' '.join(sys.argv)}"
    mlflow.log_param("command_line", command_line)
    
    # Create directory for figures and artifacts
    figure_dir = os.path.join(summary_dir, 'mlflow', experiment_name)
    os.makedirs(figure_dir, exist_ok=True)
    logging.info(f"Created figure directory: {figure_dir}")
    
    # Load data
    merged_feather_path = os.path.join(summary_dir, args.merged_feather_file)
    logging.info(f"Loading data from {merged_feather_path}")
    local_hour_adjusted_df = pd.read_feather(merged_feather_path)
    logging.info(f"Loaded dataframe with shape: {local_hour_adjusted_df.shape}")
    
    # Apply filters
    if args.filters:
        logging.info("Applying filters...")
        filter_function_pairs = args.filters.split(';')
        applied_filters = []
        for filter_function_pair in filter_function_pairs:
            filter_parts = filter_function_pair.split(',')
            filter_function_name = filter_parts[0]
            filter_params = filter_parts[1:]
            if filter_function_name in globals():
                logging.info(f"Applying filter: {filter_function_name} with parameters {filter_params}")
                local_hour_adjusted_df = globals()[filter_function_name](local_hour_adjusted_df, *filter_params)
                applied_filters.append(f"{filter_function_name}({','.join(filter_params)})")
            else:
                logging.warning(f"Filter function {filter_function_name} not found. Skipping.")
    
        # Log applied filters
        mlflow.log_param("applied_filters", "; ".join(applied_filters))
        logging.info(f"Dataframe shape after applying filters: {local_hour_adjusted_df.shape}")
        mlflow.log_param(f"data_shape_after_filers", local_hour_adjusted_df.shape)
    else:
        mlflow.log_param("applied_filters", "None")
        logging.info("No filters applied")
    
    
    # Load feature list from Excel file
    logging.info("Loading feature list...")
    df_daily_vars = pd.read_excel('/home/jguo/research/hw_global/Data/hourlyDataSchema.xlsx')  # Replace with your file path
    
    # Select features based on columns specified in arguments
    daily_vars = df_daily_vars.loc[df_daily_vars[args.feature_column] == 'Y', 'Variable']
    daily_var_lst = daily_vars.tolist()
    
    delta_vars = df_daily_vars.loc[df_daily_vars[args.delta_column] == 'Y', 'Variable']
    delta_var_lst = delta_vars.tolist()
    
    hw_nohw_diff_vars = df_daily_vars.loc[df_daily_vars[args.hw_nohw_diff_column] == 'Y', 'Variable']
    daily_var_lst.extend([f"hw_nohw_diff_{var}" for var in hw_nohw_diff_vars])  # Add HW-NoHW diff features
    
    double_diff_vars = df_daily_vars.loc[df_daily_vars[args.double_diff_column] == 'Y', 'Variable']
    daily_var_lst.extend([f"Double_Differencing_{var}" for var in double_diff_vars])  # Add Double Differencing features
    
    
    # Calculate delta variables based on delta_mode argument
    if args.delta_mode in ["include", "only"]:
        logging.info("Calculating delta variables...")
        for var in delta_var_lst:
            var_U = f"{var}_U"
            var_R = f"{var}_R"
            delta_var = f"delta_{var}"
            if var_U in local_hour_adjusted_df.columns and var_R in local_hour_adjusted_df.columns:
                local_hour_adjusted_df[delta_var] = local_hour_adjusted_df[var_U] - local_hour_adjusted_df[var_R]
            else:
                logging.warning(f"{var_U} or {var_R} not found in dataframe columns.")
    
        if args.delta_mode == "only":
            logging.info("Using only delta variables...")
            daily_var_lst = [f"delta_{var}" for var in delta_var_lst]
        else:  # "include"
            logging.info("Including delta variables...")
            daily_var_lst += [f"delta_{var}" for var in delta_var_lst]
    else:
        logging.info("Using original variables only...")
    
    # Calculate Double Differencing variables
    logging.info("Calculating Double Differencing variables...")
    for var in double_diff_vars:
        var_U = f"hw_nohw_diff_{var}_U"
        var_R = f"hw_nohw_diff_{var}_R"
        double_diff_var = f"Double_Differencing_{var}"
        if var_U in local_hour_adjusted_df.columns and var_R in local_hour_adjusted_df.columns:
            local_hour_adjusted_df[double_diff_var] = local_hour_adjusted_df[var_U] - local_hour_adjusted_df[var_R]
        else:
            logging.warning(f"{var_U} or {var_R} not found in dataframe columns.")
    
    logging.info(f"Final feature list: {daily_var_lst}")
    
    # Save and log df_daily_vars and daily_var_lst
    df_daily_vars_path = os.path.join(figure_dir, 'hourlyDataSchema.xlsx')
    df_daily_vars.to_excel(df_daily_vars_path, index=False)
    mlflow.log_artifact(df_daily_vars_path)
    logging.info(f"Saved df_daily_vars to {df_daily_vars_path}")
    
    daily_var_lst_path = os.path.join(figure_dir, 'daily_var_lst.txt')
    with open(daily_var_lst_path, 'w') as f:
        for var in daily_var_lst:
            f.write(f"{var}\n")
    mlflow.log_artifact(daily_var_lst_path)
    logging.info(f"Saved daily_var_lst to {daily_var_lst_path}")
    
    # Define day and night masks and separate data based on time_period argument
    logging.info("Defining day and night masks...")
    daytime_mask = local_hour_adjusted_df['local_hour'].between(7, 16)
    nighttime_mask = (
            local_hour_adjusted_df['local_hour'].between(20, 24) | local_hour_adjusted_df['local_hour'].between(0, 6))
    
    local_hour_adjusted_df['date'] = pd.to_datetime(local_hour_adjusted_df['local_time']).dt.date
    
    logging.info(f"Separating {args.time_period} data...")
    if args.time_period == "day":
        uhi_diff = local_hour_adjusted_df[daytime_mask]
    elif args.time_period == "night":
        uhi_diff = local_hour_adjusted_df[nighttime_mask]
    elif args.time_period == "hourly":
        uhi_diff = local_hour_adjusted_df  # No mask applied
        logging.info("Using all data without applying any time mask for 'hourly' time period.")
    
    # Calculate daily average if daily_freq argument is set
    if args.daily_freq:
        logging.info("Calculating daily average...")
        # Subset the columns before grouping
        uhi_diff = uhi_diff[daily_var_lst + ['UHI_diff', 'lat', 'lon', 'date']]
        # Now perform the grouping and aggregation
        uhi_diff = uhi_diff.groupby(['lat', 'lon', 'date']).mean().reset_index()
    
    X = uhi_diff[daily_var_lst]
    y = uhi_diff['UHI_diff']
    logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Get feature groups
    feature_groups = [get_feature_group(feature) for feature in daily_var_lst]
    
    # Log mean UHI_diff
    mean_uhi_diff = y.mean()
    mlflow.log_metric(f"{args.time_period}_mean_uhi_diff", mean_uhi_diff)
    logging.info(f"Logged mean value of UHI_diff for {args.time_period} time period: {mean_uhi_diff:.4f}")
    
    clear_gpu_memory()
    
    # Initialize CatBoost model
    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        task_type='GPU',  # Use GPU for training
        devices='0',  # Specify GPU device index
        verbose=False  # Suppress verbose output
    )
    
    # No feature selection, directly train on full dataset
    logging.info("Training the final model on the full dataset...")
    clear_gpu_memory()
    model.fit(X, y, verbose=False)
    mlflow.log_param(f"{args.time_period}_iterations", model.get_param('iterations'))
    mlflow.log_param(f"{args.time_period}_learning_rate", model.get_param('learning_rate'))
    mlflow.log_param(f"{args.time_period}_depth", model.get_param('depth'))
    mlflow.log_param(f"{args.time_period}_optimal_num_features", len(daily_var_lst))
    mlflow.log_param(f"{args.time_period}_selected_features", ", ".join(daily_var_lst))
    
    # Evaluate the model and log metrics
    y_pred = model.predict(X)
    
    train_rmse = mean_squared_error(y, y_pred, squared=False)
    train_r2 = r2_score(y, y_pred)
    
    mlflow.log_metric(f"{args.time_period}_whole_rmse", train_rmse)
    mlflow.log_metric(f"{args.time_period}_whole_r2", train_r2)
    
    logging.info(f"Model {args.time_period} metrics:")
    logging.info(f"Whole RMSE: {train_rmse:.4f}")
    logging.info(f"Whole R2: {train_r2:.4f}")
    
    # Log the trained model
    logging.info("Logging the trained model...")
    mlflow.catboost.log_model(model, f"{args.time_period}_model")
    
    # Create a pool for feature importance calculations
    full_pool = Pool(X, y)
    
    # Calculate SHAP values
    logging.info("Calculating SHAP values...")
    shap_values = model.get_feature_importance(full_pool, type='ShapValues')[:, :-1]
    feature_names = X.columns.tolist()
    
    # Ensure that the length of shap_values matches the number of rows in uhi_diff
    if shap_values.shape[0] != uhi_diff.shape[0]:
        raise ValueError("The number of SHAP values does not match the number of rows in uhi_diff.")
    
    # Extract additional columns for further analysis
    additional_columns = ['global_event_ID', 'lon', 'lat', 'time', 'KGClass', 'KGMajorClass', 'UHI_diff']
    additional_columns = [col for col in additional_columns if col in uhi_diff.columns]
    shap_df_additional_columns = uhi_diff[additional_columns].reset_index(drop=True)
    
    # Calculate estimation error
    y_pred = model.predict(X)
    estimation_error = y_pred - shap_df_additional_columns['UHI_diff']
    shap_df_additional_columns['Estimation_Error'] = estimation_error
    
    # Save SHAP values and additional columns if needed
    shap_values_path = os.path.join(figure_dir, 'shap_values.npy')
    np.save(shap_values_path, shap_values)
    mlflow.log_artifact(shap_values_path)
    logging.info(f"Saved SHAP values to {shap_values_path}")
    
    # Combine SHAP values with additional columns
    shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
    combined_df = pd.concat([shap_values_df, shap_df_additional_columns], axis=1)
    
    # Save the combined DataFrame as a Feather file
    combined_feather_path = os.path.join(figure_dir, 'shap_values_with_additional_columns.feather')
    combined_df.reset_index(drop=True).to_feather(combined_feather_path)
    mlflow.log_artifact(combined_feather_path)
    logging.info(f"Saved combined SHAP values and additional columns to {combined_feather_path}")
    
    # Now process SHAP values
    logging.info("Processing SHAP values and generating plots...")
    process_shap_values(shap_values, feature_names, X, shap_df_additional_columns, args.time_period, df_daily_vars, figure_dir, exclude_features=args.exclude_features)
    
    # Process SHAP values by KGMajorClass
    process_shap_values_by_kg_major_class(shap_values, feature_names, X, shap_df_additional_columns, args.time_period, df_daily_vars, figure_dir, exclude_features=args.exclude_features)
    
    logging.info("Script execution completed.")
    mlflow.end_run()

if __name__ == "__main__":
    main()
