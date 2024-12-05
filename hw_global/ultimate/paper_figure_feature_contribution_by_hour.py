import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import logging
import mlflow
import seaborn as sns


def report_shap_contribution_from_feather(shap_values_feather_path, output_dir, output_feature_group, output_pivot, output_feature):
    """
    Reports SHAP value contributions from each feature group and feature by hour and by KGMajorClass
    using data from a Feather file.
    
    Args:
        shap_values_feather_path (str): Path to the shap_values_with_additional_columns.feather file.
        output_dir (str): Directory to save the output files.
        output_feature_group (str): Path to save the aggregated feature group data.
        output_pivot (str): Path to save the pivoted feature group data.
        output_feature (str): Path to save the aggregated feature data.
    """
    # Load the shap values with additional columns
    shap_df = pd.read_feather(shap_values_feather_path)
    
    # Step 1: Drop specified columns
    columns_to_drop = ['global_event_ID', 'lon', 'lat', 'time', 'KGClass']
    shap_df = shap_df.drop(columns=columns_to_drop, errors='ignore')
    logging.info(f"Dropped columns: {columns_to_drop}")
    
    # Step 2: Group by 'local_hour' and 'KGMajorClass' and sum all shap value columns
    group_cols = ['local_hour', 'KGMajorClass']

    # Identify SHAP value columns: those ending with '_shap' and not in group_cols
    shap_cols = [col for col in shap_df.columns if col.endswith('_shap') and col not in [f"{gc}_shap" for gc in group_cols]]
    
    if not shap_cols:
        logging.warning("No SHAP columns found after excluding group-related columns.")
    
    # Group and sum the shap value columns
    df_grouped = shap_df.groupby(group_cols)[shap_cols].sum().reset_index()
    logging.info("Grouped by 'local_hour' and 'KGMajorClass' and summed SHAP value columns.")
    
    # Step 3: Prepare feature group mapping with base feature names
    # Create a mapping from SHAP columns to base feature names
    col_mapping = {col: col.replace('_shap', '') for col in shap_cols}

    # Exclude certain columns if necessary (adjust as per your requirements)
    exclude_cols = ['UHI_diff_shap', 'Estimation_Error_shap']
    feature_group_mapping = {}
    # for col in shap_cols:
    #     if col not in exclude_cols:
    #         base_col = col_mapping[col]
    #         group = get_feature_groups(base_col)
    #         feature_group_mapping[base_col] = group

    # # Create a DataFrame for feature groups
    # feature_groups = pd.DataFrame({
    #     'Feature': list(feature_group_mapping.keys()),
    #     'Feature Group': list(feature_group_mapping.values())
    # })
    feature_names = [col.replace('_shap', '') for col in shap_cols]
    feature_groups = get_feature_groups(feature_names)
    
    # Convert feature_groups dictionary to a DataFrame
    feature_groups_df = pd.DataFrame(
        list(feature_groups.items()), 
        columns=['Feature', 'Feature Group']
    )
    
    # Melt the dataframe to long format with base feature names
    df_melted = df_grouped.melt(
        id_vars=group_cols, 
        value_vars=shap_cols,
        var_name='Feature', 
        value_name='Value'
    )
    
    # Replace SHAP column names with base feature names
    df_melted['Feature'] = df_melted['Feature'].map(col_mapping)
    
    # Merge to get feature groups
    df_melted = df_melted.merge(feature_groups_df, on='Feature', how='left')
    
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
    )
    
    # Reset index and ensure column names are unique
    df_pivot = df_pivot.reset_index()
    
    # Step 4: Save the processed dataframes as Feather files
    df_feature_group.to_feather(output_feature_group)
    logging.info(f"Saved feature group dataframe to Feather file at {output_feature_group}")
    
    df_pivot.to_feather(output_pivot)
    logging.info(f"Saved pivoted dataframe to Feather file at {output_pivot}")
    
    df_feature.to_feather(output_feature)
    logging.info(f"Saved per-feature dataframe to Feather file at {output_feature}")
    
    return df_feature_group, df_feature

def load_feature_values(shap_values_feather_path):
    """
    Loads feature values from the shap_values_with_additional_columns.feather file.

    Args:
        shap_values_feather_path (str): Path to the shap_values_with_additional_columns.feather file.

    Returns:
        pd.DataFrame: DataFrame containing feature values, aligned with SHAP values.
    """
    # Load the shap values with additional columns
    shap_df = pd.read_feather(shap_values_feather_path)

    # Identify feature columns: columns not in exclude list and not ending with '_shap'
    exclude_cols = [
        'global_event_ID', 'lon', 'lat', 'time', 'KGClass', 'KGMajorClass', 
        'local_hour', 'UHI_diff', 'Estimation_Error'
    ]
    feature_cols = [
        col for col in shap_df.columns 
        if col not in exclude_cols and not col.endswith('_shap')
    ]

    # Extract the required columns
    feature_values_df = shap_df[
        [
            'global_event_ID', 'lon', 'lat', 'time', 'KGClass', 
            'KGMajorClass', 'local_hour'
        ] + feature_cols
    ]

    # Melt the feature values dataframe to long format
    feature_values_melted = feature_values_df.melt(
        id_vars=[
            'global_event_ID', 'lon', 'lat', 'time', 'KGClass', 
            'KGMajorClass', 'local_hour'
        ],
        value_vars=feature_cols,
        var_name='Feature',
        value_name='FeatureValue'
    )

    return feature_values_melted

def get_top_features(df, top_n):
    """
    Gets the list of top features based on total contribution.

    Args:
        df (pd.DataFrame): The DataFrame containing features.
        top_n (int): The number of top features to select.

    Returns:
        list: List of top feature names.
    """
    feature_totals = df.groupby('Feature')['Value'].sum()
    top_features_list = feature_totals.sort_values(ascending=False).head(top_n).index.tolist()
    return top_features_list

def get_feature_groups(feature_names):
    """
    Assign features to groups based on specified rules.

    Args:
        feature_names (list): List of feature names.

    Returns:
        dict: Mapping from feature names to group names.
    """
    prefixes = ('delta_', 'hw_nohw_diff_', 'Double_Differencing_')
    feature_groups = {}
    for feature in feature_names:
        group = feature
        for prefix in prefixes:
            if feature.startswith(prefix):
                group = feature[len(prefix):]
                break
        # If feature does not start with any prefix, it is its own group, but name the group feature + "Level"
        if group == feature:
            group = feature + "_Level"
        feature_groups[feature] = group
    return feature_groups

def plot_shap_stacked_bar(shap_df, title, output_path, color_mapping=None, return_fig=False):
    """
    Plots a standalone SHAP stacked bar plot (all features) with a total SHAP value curve.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    if color_mapping:
        # Ensure columns are sorted to match color mapping
        sorted_columns = sorted(shap_df.columns, key=lambda x: x)
        colors = [color_mapping.get(feature, '#333333') for feature in sorted_columns]
        shap_df = shap_df[sorted_columns]
        shap_df.plot(kind='bar', stacked=True, color=colors, ax=ax)
    else:
        # Default colormap if no color_mapping is provided
        shap_df.plot(kind='bar', stacked=True, colormap='tab20', ax=ax)

    # Calculate total SHAP values for each hour
    total_shap = shap_df.sum(axis=1)

    # Plot the total SHAP values as a curve on a secondary y-axis
    ax2 = ax.twinx()
    total_shap.plot(kind='line', color='black', marker='o', linewidth=2, ax=ax2, label='Total SHAP')
    ax2.set_ylabel('Total SHAP Value')

    # Combine legends from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

    plt.title(title)
    plt.xlabel('Hour of Day')
    ax.set_ylabel('SHAP Value Contribution')
    plt.tight_layout()

    if return_fig:
        return fig, ax
    else:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Standalone SHAP stacked bar plot with total curve saved at '{output_path}'.")

def plot_shap_and_feature_values_for_group(shap_df, feature_values_df, group_name, output_dir, kg_class, color_mapping):
    """
    Plots SHAP value contributions and feature group's values side by side.
    Saves each plot as a separate image file with the legend at the bottom.
    Calls the plot_shap_stacked_bar function for SHAP plotting.
    """
    import matplotlib.pyplot as plt
    import os

    # Check if data is available
    if shap_df.empty or feature_values_df.empty:
        print(f"No data available for group '{group_name}' in KGMajorClass '{kg_class}'. Skipping.")
        return

    # Create a subdirectory for the current feature group if it doesn't exist
    group_dir = os.path.join(output_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)

    # Prepare the output paths
    output_filename = f'shap_and_feature_values_{group_name}_{kg_class}.png'
    output_path = os.path.join(group_dir, output_filename)

    shap_output_filename = f'shap_contributions_{group_name}_{kg_class}.png'
    shap_output_path = os.path.join(group_dir, shap_output_filename)

    # Plot SHAP contributions using plot_shap_stacked_bar (standalone plot)
    plot_shap_stacked_bar(
        shap_df=shap_df,
        title=f'SHAP Value Contributions - {group_name} - KGMajorClass {kg_class}',
        output_path=shap_output_path,
        color_mapping=color_mapping,
        return_fig=False
    )

    # Plot SHAP contributions and feature values for combined plot
    fig_combined, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8), sharex=False)

    # Define colors based on the color mapping
    shap_colors = [color_mapping.get(feature, '#333333') for feature in shap_df.columns]
    feature_colors = [color_mapping.get(feature, '#333333') for feature in feature_values_df.columns]

    # Plot SHAP contributions on axes[0]
    shap_df.plot(kind='bar', stacked=True, ax=axes[0], color=shap_colors)
    axes[0].set_title('SHAP Value Contributions')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Contribution')

    # Calculate total SHAP values for each hour
    total_shap = shap_df.sum(axis=1)

    # Plot the total SHAP values as a curve on a secondary y-axis
    ax2 = axes[0].twinx()
    total_shap.plot(kind='line', color='black', marker='o', linewidth=2, ax=ax2, label='Total SHAP')
    ax2.set_ylabel('Total SHAP Value')

    # Combine legends from both axes
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[0].legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

    # Plot feature values on axes[1]
    feature_values_df.plot(ax=axes[1], color=feature_colors)
    axes[1].set_title(f'Feature Values - Group: {group_name}')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Feature Value')
    axes[1].axhline(0, linestyle='--', color='lightgray', linewidth=1)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

    # Adjust layout and save the figure
    title = f'ΔUHI Contribution and Feature Values by Hour - {group_name} - KGMajorClass {kg_class}'
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as '{output_path}' for group '{group_name}' and KGMajorClass '{kg_class}'.")

def plot_shap_and_feature_values(df_feature, feature_values_melted, kg_classes, output_dir):
    """
    Plots SHAP value contributions and feature group's values side by side.
    Generates plots for global data and each KGMajorClass filtered data.
    Saves each plot as a separate image file with the legend at the bottom.
    Organizes outputs into the global subdirectory and separate subdirectories for each KGMajorClass.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Prepare a list of all feature groups
    feature_names = df_feature['Feature'].unique().tolist()
    feature_groups = get_feature_groups(feature_names)
    unique_groups = set(feature_groups.values())
    
    # Create a color palette
    palette = sns.color_palette('tab20', n_colors=len(feature_names))
    color_mapping = dict(zip(sorted(feature_names), palette))
    
    # Create a global subdirectory
    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)
    
    # Generate and save the standalone SHAP stacked bar plot for global data (all features)
    print("Generating global standalone SHAP stacked bar plot (all features)...")
    shap_plot_df_global = df_feature.pivot_table(
        index='local_hour',
        columns='Feature',
        values='Value',
        fill_value=0
    ).reset_index()
    shap_plot_df_global.set_index('local_hour', inplace=True)
    
    # Plot and save the standalone SHAP stacked bar plot for global data
    plot_shap_stacked_bar(
        shap_df=shap_plot_df_global,
        title='Global SHAP Value Contributions (All Features)',
        output_path=os.path.join(global_dir, 'global_shap_stacked_bar_all_features.png'),
        color_mapping=color_mapping
    )
    
    # Generate side-by-side plots for each feature group in global data
    print("Generating global plots for each feature group...")
    for group_name in unique_groups:
        # Filter df_feature for features in the current group
        group_features = [f for f, g in feature_groups.items() if g == group_name]
        
        shap_plot_df_group = df_feature[df_feature['Feature'].isin(group_features)]
        shap_plot_df_group = shap_plot_df_group.groupby(['local_hour', 'Feature'])['Value'].sum().reset_index()
        shap_plot_df_group = shap_plot_df_group.pivot_table(
            index='local_hour',
            columns='Feature',
            values='Value',
            fill_value=0
        ).reset_index()
        shap_plot_df_group.set_index('local_hour', inplace=True)
    
        feature_values_plot_df_group = feature_values_melted[feature_values_melted['Feature'].isin(group_features)]
        feature_values_plot_df_group = feature_values_plot_df_group.groupby(['local_hour', 'Feature'])['FeatureValue'].mean().reset_index()
        feature_values_plot_df_group = feature_values_plot_df_group.pivot_table(
            index='local_hour',
            columns='Feature',
            values='FeatureValue',
            fill_value=0
        ).reset_index()
        feature_values_plot_df_group.set_index('local_hour', inplace=True)
    
        plot_shap_and_feature_values_for_group(
            shap_df=shap_plot_df_group,
            feature_values_df=feature_values_plot_df_group,
            group_name=group_name,
            output_dir=global_dir,  # Save under the global directory
            kg_class='global',      # Indicate global in the plot filenames
            color_mapping=color_mapping
        )
    
    # Plot for each KGMajorClass
    for kg_class in kg_classes:
        print(f"Generating plots for KGMajorClass '{kg_class}'...")
        # Create subdirectory for the current kg_class
        kg_class_dir = os.path.join(output_dir, kg_class)
        os.makedirs(kg_class_dir, exist_ok=True)
    
        # Filter data for the current kg_class
        df_feature_subset = df_feature[df_feature['KGMajorClass'] == kg_class]
        feature_values_subset = feature_values_melted[feature_values_melted['KGMajorClass'] == kg_class]
        
        # Check if there's data for the current kg_class
        if df_feature_subset.empty:
            print(f"No data for KGMajorClass '{kg_class}'. Skipping.")
            continue
        
        # Generate and save the standalone SHAP stacked bar plot for the current kg_class (all features)
        print(f"Generating standalone SHAP stacked bar plot for KGMajorClass '{kg_class}'...")
        shap_plot_df_kg = df_feature_subset.pivot_table(
            index='local_hour',
            columns='Feature',
            values='Value',
            fill_value=0
        ).reset_index()
        shap_plot_df_kg.set_index('local_hour', inplace=True)
        
        plot_shap_stacked_bar(
            shap_df=shap_plot_df_kg,
            title=f'SHAP Value Contributions (All Features) - KGMajorClass {kg_class}',
            output_path=os.path.join(kg_class_dir, f'{kg_class}_shap_stacked_bar_all_features.png'),
            color_mapping=color_mapping
        )
        
        # Generate side-by-side plots for each feature group
        for group_name in unique_groups:
            # Filter df_feature_subset for features in the current group
            group_features = [f for f, g in feature_groups.items() if g == group_name]
    
            shap_plot_df = df_feature_subset[df_feature_subset['Feature'].isin(group_features)]
            shap_plot_df = shap_plot_df.groupby(['local_hour', 'Feature'])['Value'].sum().reset_index()
            shap_plot_df = shap_plot_df.pivot_table(
                index='local_hour',
                columns='Feature',
                values='Value',
                fill_value=0
            ).reset_index()
            shap_plot_df.set_index('local_hour', inplace=True)
    
            feature_values_plot_df = feature_values_subset[feature_values_subset['Feature'].isin(group_features)]
            feature_values_plot_df = feature_values_plot_df.groupby(['local_hour', 'Feature'])['FeatureValue'].mean().reset_index()
            feature_values_plot_df = feature_values_plot_df.pivot_table(
                index='local_hour',
                columns='Feature',
                values='FeatureValue',
                fill_value=0
            ).reset_index()
            feature_values_plot_df.set_index('local_hour', inplace=True)
    
            plot_shap_and_feature_values_for_group(
                shap_df=shap_plot_df,
                feature_values_df=feature_values_plot_df,
                group_name=group_name,
                output_dir=kg_class_dir,   # Save under the kg_class directory
                kg_class=kg_class,        # Indicate kg_class in the plot filenames
                color_mapping=color_mapping
            )

# Define the function with 'Value' instead of 'Importance'
def plot_feature_group_stacked_bar(df, group_by_column, output_path, title):
    """
    Plots a stacked bar chart of feature group contributions with total SHAP value line.

    Parameters:
    - df: DataFrame containing 'local_hour', 'Feature Group', and 'Value'.
    - group_by_column: Column name to group by ('local_hour').
    - output_path: Path to save the output plot.
    - title: Title of the plot.
    """
    # Pivot the data to have feature groups as columns
    pivot_df = df.pivot_table(
        index=group_by_column,
        columns='Feature Group',
        values='Value',
        aggfunc='sum',
        fill_value=0  # Replace NaNs with zeros
    )

    # Sort the index if necessary
    pivot_df = pivot_df.sort_index()

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot stacked bars on primary axis
    pivot_df.plot(
        kind='bar',
        stacked=True,
        ax=ax1
    )

    # Calculate and plot total SHAP values on secondary axis
    ax2 = ax1.twinx()
    total_shap = pivot_df.sum(axis=1)
    total_shap.plot(color='black', marker='o', linewidth=2, 
                    ax=ax2, label='Total SHAP')
    ax2.set_ylabel('Total SHAP Value')

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, 
              bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    ax1.set_xlabel(group_by_column.replace('_', ' ').title())
    ax1.set_ylabel('SHAP Value Contribution')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    import mlflow
    import logging
    import os

    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Report and plot SHAP value contributions by feature group and hour."
    )

    # Add command-line arguments
    parser.add_argument(
        '--experiment-name',
        type=str,
        required=True,
        help='Name of the MLflow experiment to process.'
    )
    parser.add_argument(
        '--top-features',
        type=int,
        default=None,
        help='Number of top features to plot. If None, plot all features.'
    )

    # Parse the arguments
    args = parser.parse_args()

    mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")

    # Extract arguments
    experiment_name = args.experiment_name
    top_features = args.top_features

    # Process a single experiment by loading the necessary data
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.error(f"Experiment '{experiment_name}' not found.")
        return

    experiment_id = experiment.experiment_id

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1
    )
    if len(runs) == 0:
        logging.error(f"No runs found in experiment '{experiment_name}'. Please check the experiment name and make sure it contains runs.")
        return

    run = runs.iloc[0]
    run_id = run.run_id

    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    # Replace the placeholder with the actual path in your environment
    artifact_uri = artifact_uri.replace(
        "mlflow-artifacts:",
        "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts"
    )

    shap_values_feather_path = os.path.join(artifact_uri, "shap_values_with_additional_columns.feather")
    output_dir = os.path.join(artifact_uri, '24_hourly_plot')

    os.makedirs(output_dir, exist_ok=True)

    output_feature_group = os.path.join(output_dir, 'shap_feature_group.feather')
    output_pivot = os.path.join(output_dir, 'shap_pivot.feather')
    output_feature = os.path.join(output_dir, 'shap_feature.feather')

    # Report SHAP contribution
    df_feature_group, df_feature = report_shap_contribution_from_feather(
        shap_values_feather_path,
        output_dir,
        output_feature_group,
        output_pivot,
        output_feature
    )
    print("SHAP contribution by hour generated.")

    # Load the feature values from shap_values_feather_path
    feature_values_melted = load_feature_values(shap_values_feather_path)

    # Optionally select top features
    if top_features is not None:
        # Get top features based on total SHAP values
        top_features_list = get_top_features(df_feature, top_features)
        df_feature = df_feature[df_feature['Feature'].isin(top_features_list)]
        feature_values_melted = feature_values_melted[feature_values_melted['Feature'].isin(top_features_list)]

    # Generate plots for global data and each KGMajorClass
    kg_major_classes = df_feature['KGMajorClass'].unique().tolist()

    plot_shap_and_feature_values(
        df_feature,
        feature_values_melted,
        kg_major_classes,
        output_dir
    )

    # Generate stacked bar plot for global data grouped by 'local_hour'
    output_path_global = os.path.join(output_dir, 'feature_group_contribution_by_hour_global.png')
    plot_feature_group_stacked_bar(
        df_feature_group,
        group_by_column='local_hour',
        output_path=output_path_global,
        title='Global Feature Group Contribution by Hour'
    )

    # Generate stacked bar plot for each KGMajorClass grouped by 'local_hour'
    kg_classes = df_feature_group['KGMajorClass'].unique()
    for kg_major_class in kg_classes:
        kg_hourly_data = df_feature_group[df_feature_group['KGMajorClass'] == kg_major_class]
        output_path_kg = os.path.join(output_dir, f'feature_group_contribution_by_hour_{kg_major_class}.png')
        plot_feature_group_stacked_bar(
            kg_hourly_data,
            group_by_column='local_hour',
            output_path=output_path_kg,
            title=f'Feature Group Contribution by Hour for {kg_major_class}'
        )

if __name__ == "__main__":
    main()
