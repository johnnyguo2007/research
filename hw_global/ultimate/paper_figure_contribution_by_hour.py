import pandas as pd
import matplotlib.pyplot as plt
from hourly_kg_model import get_feature_group  # Ensure this import path is correct

import os
import sys
import argparse
import logging

def load_data(feather_path):
    """Loads data from a Feather file."""
    df = pd.read_feather(feather_path)
    return df

def add_feature_group(df):
    """Adds a 'Feature Group' column to the dataframe."""
    df['Feature Group'] = df['Feature'].apply(get_feature_group)
    return df

def calculate_percentage_contribution(df):
    """
    Calculates the percentage contribution of each feature group per hour.
    
    Assumes there is an 'hour' column in the dataframe representing hours 0-23.
    """
    # Group by 'hour' and 'Feature Group' and count occurrences
    grouped = df.groupby(['hour', 'Feature Group']).size().unstack(fill_value=0)
    
    # Calculate percentage
    percentage = grouped.divide(grouped.sum(axis=1), axis=0) * 100
    return percentage


def report_shap_contribution_from_feather(local_hour_adjusted_df_path, shap_values_feather_path, output_dir):
    """
    Reports SHAP value contributions from each feature group by hour and by KGMajorClass
    using data from a Feather file and a local_hour_adjusted_df file.
    
    Args:
        local_hour_adjusted_df_path (str): Path to the local_hour_adjusted_df file.
        shap_values_feather_path (str): Path to the shap_values_with_additional_columns.feather file.
        output_dir (str): Directory to save the output CSV file.
    """
    # Load the shap values with additional columns
    shap_df = pd.read_feather(shap_values_feather_path)
    
    # Load the local_hour_adjusted_df
    local_hour_adjusted_df = pd.read_feather(local_hour_adjusted_df_path)
    
    # Replace the local_hour in shap_df with the one from local_hour_adjusted_df
    shap_df = shap_df.drop(columns=['local_hour'])
    shap_df['local_hour'] = local_hour_adjusted_df['local_hour']
    
    # Add feature groups to the dataframe
    # Assuming you want to categorize the columns themselves
    feature_columns = shap_df.columns.difference(['local_hour', 'global_event_ID', 'lon', 'lat', 'time', 'KGClass', 'KGMajorClass', 'UHI_diff', 'Estimation_Error'])
    feature_groups = {col: get_feature_group(col) for col in feature_columns}
    
    # Create a new DataFrame to store feature group contributions
    feature_group_contributions = pd.DataFrame()

    for feature, group in feature_groups.items():
        temp_df = shap_df[['local_hour', feature]].copy()
        temp_df['Feature Group'] = group
        temp_df['Contribution'] = temp_df[feature]
        temp_df['Group_Type'] = 'Hour'
        temp_df = temp_df.drop(columns=[feature])
        feature_group_contributions = pd.concat([feature_group_contributions, temp_df], ignore_index=True)

    # Report by Hour
    hourly_contribution = feature_group_contributions[feature_group_contributions['Group_Type'] == 'Hour'].reset_index(drop=True)
    
    # Report by KGMajorClass
    feature_group_contributions['Group_Type'] = 'KGMajorClass'
    kg_contribution = feature_group_contributions[feature_group_contributions['Group_Type'] == 'KGMajorClass'].reset_index(drop=True)
    
    # Combine both contributions into a single DataFrame
    combined_contribution = pd.concat([hourly_contribution, kg_contribution], ignore_index=True)
    
    # Save the combined contributions to a single CSV file
    combined_csv_path = os.path.join(output_dir, 'shap_contribution_combined.csv')
    combined_contribution.to_csv(combined_csv_path, index=False)
    logging.info(f"Saved combined SHAP contribution to {combined_csv_path}")


def plot_stacked_bar(percentage_df, output_path=None):
    """Plots a stacked bar chart of percentage contributions per hour."""
    ax = percentage_df.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='tab20')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Percentage Contribution')
    plt.title('Feature Group Contribution by Hour')
    plt.legend(title='Feature Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def main():
    # Paths to the Feather files and output directory
    shap_values_feather_path = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/Hourly_kg_model_Hourly_HW98_no_filter/shap_values_with_additional_columns.feather'
    local_hour_adjusted_df_path = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/updated_local_hour_adjusted_variables_HW98.feather'
    output_dir = './'  # Update this path as needed

    # Report SHAP contribution
    report_shap_contribution_from_feather(local_hour_adjusted_df_path, shap_values_feather_path, output_dir)
    print("SHAP contribution report generated.")

    # Load the combined contribution CSV
    combined_csv_path = os.path.join(output_dir, 'shap_contribution_combined.csv')
    combined_contribution_df = pd.read_csv(combined_csv_path)

    # Filter for hourly contributions
    hourly_contribution_df = combined_contribution_df[combined_contribution_df['Group_Type'] == 'Hour']

    # Pivot the DataFrame for plotting
    percentage_df = hourly_contribution_df.pivot(index='Group_Value', columns='Feature Group', values='value_column')  # Replace 'value_column' with the actual column name for values

    # Plot stacked bar chart
    plot_stacked_bar(percentage_df, output_path='feature_group_contribution_by_hour.png')
    print("Stacked bar chart saved as 'feature_group_contribution_by_hour.png'.")

if __name__ == "__main__":
    main()
