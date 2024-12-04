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


def report_shap_contribution_from_feather(local_hour_adjusted_df_path, shap_values_feather_path, output_dir, output_feature_group, output_pivot, output_feature):
    """
    Reports SHAP value contributions from each feature group and feature by hour and by KGMajorClass
    using data from a Feather file and a local_hour_adjusted_df file.
    
    Args:
        local_hour_adjusted_df_path (str): Path to the local_hour_adjusted_df file.
        shap_values_feather_path (str): Path to the shap_values_with_additional_columns.feather file.
        output_dir (str): Directory to save the output CSV file.
        output_feature_group (str): Path to save the aggregated feature group data.
        output_pivot (str): Path to save the pivoted feature group data.
        output_feature (str): Path to save the aggregated feature data.
    """
    # Load the shap values with additional columns
    shap_df = pd.read_feather(shap_values_feather_path)
    
    # Load the local_hour_adjusted_df
    local_hour_adjusted_df = pd.read_feather(local_hour_adjusted_df_path)
    
    # Replace the local_hour in shap_df with the one from local_hour_adjusted_df
    shap_df = shap_df.drop(columns=['local_hour'])
    shap_df['local_hour'] = local_hour_adjusted_df['local_hour']
    
    # Step 1: Drop specified columns
    columns_to_drop = ['global_event_ID', 'lon', 'lat', 'time', 'KGClass']
    shap_df = shap_df.drop(columns=columns_to_drop, errors='ignore')
    logging.info(f"Dropped columns: {columns_to_drop}")
    
    # Step 2: Group by 'local_hour' and 'KGMajorClass' and sum all numeric columns
    group_cols = ['local_hour', 'KGMajorClass']
    numeric_cols = shap_df.select_dtypes(include=['float', 'int']).columns
    numeric_cols = [col for col in numeric_cols if col != 'local_hour'] 
    df_grouped = shap_df.groupby(group_cols)[numeric_cols].sum().reset_index()
    logging.info("Grouped by 'local_hour' and 'KGMajorClass' and summed numeric columns.")
    
    # Step 3: Add columns indicating feature groups with summed feature columns
    feature_group_mapping = {col: get_feature_group(col) for col in numeric_cols if col not in ['UHI_diff', 'Estimation_Error']}
    
    # Create a DataFrame for feature groups
    feature_groups = pd.DataFrame({
        'Feature': list(feature_group_mapping.keys()),
        'Feature Group': list(feature_group_mapping.values())
    })
    
    # Melt the dataframe to long format
    df_melted = df_grouped.melt(
        id_vars=group_cols, 
        value_vars=feature_group_mapping.keys(),
        var_name='Feature', 
        value_name='Value'
    )
    
    # Merge to get feature groups
    df_melted = df_melted.merge(feature_groups, on='Feature', how='left')
    
    # Sum values by feature group
    df_feature_group = df_melted.groupby(group_cols + ['Feature Group'])['Value'].sum().reset_index()
    
    # Sum values by feature
    df_feature = df_melted.groupby(group_cols + ['Feature'])['Value'].sum().reset_index()
    
    # Pivot the dataframe for feature groups
    df_pivot = df_feature_group.pivot_table(
        index=group_cols, 
        columns='Feature Group', 
        values='Value', 
        fill_value=0
    ).reset_index()
    
    # Step 4: Save the processed dataframes as Feather files
    df_feature_group.to_feather(output_feature_group)
    logging.info(f"Saved feature group dataframe to Feather file at {output_feature_group}")
    
    df_pivot.to_feather(output_pivot)
    logging.info(f"Saved pivoted dataframe to Feather file at {output_pivot}")
    
    df_feature.to_feather(output_feature)
    logging.info(f"Saved per-feature dataframe to Feather file at {output_feature}")
    
    return df_feature_group, df_feature


def plot_stacked_bar(plot_df, title, output_path=None, total_series=None):
    """Plots a stacked bar chart of contributions per hour with an optional total value curve.

    This function plots either percentage contributions or absolute contributions based on the provided DataFrame.
    For absolute contributions, an optional total value curve can be overlaid.

    Args:
        plot_df (pd.DataFrame): DataFrame where each row corresponds to a local_hour and each column to a Feature Group or Feature,
                                 containing either percentage values that sum to 100% per local_hour or absolute values.
        title (str): Title of the plot.
        output_path (str, optional): Path to save the plot image. If None, the plot is displayed.
        total_series (pd.Series, optional): Series containing total values per hour to plot as a curve.
    """
    num_columns = len(plot_df.columns)
    if num_columns <= 10:
        colormap = 'tab10'
    elif num_columns <= 20:
        colormap = 'tab20'
    else:
        # Use a colormap that can handle many colors
        colormap = 'tab20'  # You may choose a custom colormap here

    ax = plot_df.plot(
        kind='bar', 
        stacked=True, 
        figsize=(15, 8), 
        colormap=colormap,
        width=0.8,
        ax=None
    )
    
    plt.xlabel('Hour of Day')
    ylabel = 'Percentage Contribution' if (plot_df.sum(axis=1).round(2) == 100.00).all() else 'Absolute Contribution'
    plt.ylabel(ylabel)
    plt.title(title)
    # Adjust legend placement based on the number of columns
    if num_columns <= 10:
        plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    
    # Plot the total curve
    if total_series is not None:
        ax2 = ax.twinx()
        ax2.plot(plot_df.index, total_series, color='black', marker='o', linestyle='-', label='Total Value')
        ax2.set_ylabel('Total Value')
        ax2.legend(loc='upper right')
    elif ylabel == 'Percentage Contribution':
        # For percentage plots, plot a flat line at 100%
        ax2 = ax.twinx()
        ax2.plot(plot_df.index, [100] * len(plot_df.index), color='black', linestyle='--', label='Total 100%')
        ax2.set_ylabel('Total Percentage')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def main():
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Report and plot SHAP value contributions by feature group and hour."
    )
    
    # Add command-line arguments
    parser.add_argument(
        '--local-hour-adjusted-df-path',
        type=str,
        default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/updated_local_hour_adjusted_variables_HW98.feather',
        help='Path to the local_hour_adjusted_df file.'
    )
    parser.add_argument(
        '--shap-values-feather-path',
        type=str,
        default='/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/Hourly_kg_model_Hourly_HW98_no_filter/shap_values_with_additional_columns.feather',
        help='Path to the shap_values_with_additional_columns.feather file.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/jguo/tmp/output',
        help='Directory to save the output files.'
    )
    parser.add_argument(
        '--top-features',
        type=int,
        default=None,
        help='Number of top features to plot. If None, plot all features.'
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Extract arguments
    shap_values_feather_path = args.shap_values_feather_path
    local_hour_adjusted_df_path = args.local_hour_adjusted_df_path
    output_dir = args.output_dir
    output_feature_group = os.path.join(output_dir, 'shap_feature_group.feather') 
    output_pivot = os.path.join(output_dir, 'shap_pivot.feather') 
    output_feature = os.path.join(output_dir, 'shap_feature.feather')
    top_features = args.top_features
    
    # Report SHAP contribution
    df_feature_group, df_feature = report_shap_contribution_from_feather(
        local_hour_adjusted_df_path, 
        shap_values_feather_path, 
        output_dir, 
        output_feature_group, 
        output_pivot,
        output_feature
    )
    print("SHAP contribution by hour generated.")
    
    # Define the types of plots to generate
    plot_types = ['percentage', 'absolute']
    
    # Generate plots for each KGMajorClass and each plot type
    kg_major_classes = df_feature['KGMajorClass'].unique()
    for kg_class in kg_major_classes:
        df_subset = df_feature[df_feature['KGMajorClass'] == kg_class]
        
        # Group by 'local_hour' and 'Feature'
        grouped = df_subset.groupby(['local_hour', 'Feature'])['Value'].sum().reset_index()
        
        # Pivot to have Features as columns
        pivot_df = grouped.pivot(index='local_hour', columns='Feature', values='Value').fillna(0)
        
        for plot_type in plot_types:
            if plot_type == 'percentage':
                # Calculate percentages
                plot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
                total_series = None  # Total not needed for percentage plots
            else:
                # Use absolute values
                plot_df = pivot_df
                total_series = plot_df.sum(axis=1)
    
            # Optionally select top features
            if top_features is not None:
                # Calculate total contribution of each feature over all hours
                feature_totals = plot_df.sum()
                top_features_list = feature_totals.sort_values(ascending=False).head(top_features).index
                plot_df = plot_df[top_features_list]
    
            # Define the plot title
            title = f'ΔUHI Contribution by Hour (Features) - {kg_class}' + (f' ({plot_type.capitalize()})' if plot_type == 'percentage' else f' ({plot_type.capitalize()})')
    
            # Define the output path
            output_filename = f'feature_contribution_by_hour_{kg_class}_{plot_type}.png'
            output_path = os.path.join(output_dir, output_filename)
    
            # Plot stacked bar chart for the current KGMajorClass and plot type
            plot_stacked_bar(plot_df, title, output_path=output_path, total_series=total_series if plot_type == 'absolute' else None)
            print(f"Stacked bar chart saved as '{output_path}' for KGMajorClass '{kg_class}' ({plot_type}).")
    
    # Generate total plots by aggregating across all KGMajorClasses
    total_grouped = df_feature.groupby(['local_hour', 'Feature'])['Value'].sum().reset_index()
    
    # Pivot to have Features as columns
    total_pivot_df = total_grouped.pivot(index='local_hour', columns='Feature', values='Value').fillna(0)
    
    for plot_type in plot_types:
        if plot_type == 'percentage':
            # Calculate percentages
            total_plot_df = total_pivot_df.div(total_pivot_df.sum(axis=1), axis=0) * 100
            total_series = None  # Total not needed for percentage plots
        else:
            # Use absolute values
            total_plot_df = total_pivot_df
            total_series = total_plot_df.sum(axis=1)
        
        # Optionally select top features
        if top_features is not None:
            # Calculate total contribution of each feature over all hours
            feature_totals = total_plot_df.sum()
            top_features_list = feature_totals.sort_values(ascending=False).head(top_features).index
            total_plot_df = total_plot_df[top_features_list]
        
        # Define the plot title for the total plot
        total_title = f'ΔUHI Contribution by Hour (Features) - Global' + (f' ({plot_type.capitalize()})' if plot_type == 'percentage' else f' ({plot_type.capitalize()})')
        
        # Define the output path
        total_output_filename = f'feature_contribution_by_hour_total_{plot_type}.png'
        total_output_path = os.path.join(output_dir, total_output_filename)

        # Plot total stacked bar chart
        plot_stacked_bar(total_plot_df, total_title, output_path=total_output_path, total_series=total_series if plot_type == 'absolute' else None)
        print(f"Total stacked bar chart saved as '{total_output_path}' ({plot_type}).")

if __name__ == "__main__":
    main()
