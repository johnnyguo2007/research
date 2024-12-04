import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import logging
import mlflow

from hourly_kg_model import get_feature_group  # Ensure this import path is correct

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

    # Identify SHAP value columns: those ending with '_shap'
    shap_cols = [col for col in shap_df.columns if col.endswith('_shap')]
    
    # Group and sum the shap value columns
    df_grouped = shap_df.groupby(group_cols)[shap_cols].sum().reset_index()
    logging.info("Grouped by 'local_hour' and 'KGMajorClass' and summed SHAP value columns.")
    
    # Step 3: Add columns indicating feature groups with summed feature columns
    feature_group_mapping = {col: get_feature_group(col) for col in shap_cols if col not in ['UHI_diff_shap', 'Estimation_Error_shap']}
    
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

def prepare_plot_df(df, feature_col, value_col, group_cols):
    """
    Prepares the DataFrame for plotting by grouping and pivoting.

    Args:
        df (pd.DataFrame): The DataFrame to prepare.
        feature_col (str): The name of the feature column.
        value_col (str): The name of the value column.
        group_cols (list): The columns to group by.

    Returns:
        pd.DataFrame: The prepared DataFrame for plotting.
    """
    # Group by specified columns and sum values
    grouped = df.groupby(group_cols + [feature_col])[value_col].sum().reset_index()

    # Pivot to have features as columns
    pivot_df = grouped.pivot(
        index=group_cols[0], columns=feature_col, values=value_col
    ).fillna(0)

    return pivot_df

def get_top_features(df, top_n):
    """
    Gets the list of top features based on total contribution.

    Args:
        df (pd.DataFrame): The DataFrame containing features.
        top_n (int): The number of top features to select.

    Returns:
        list: List of top feature names.
    """
    feature_totals = df.sum()
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

def plot_shap_and_feature_values_for_group(shap_df, feature_values_df, group_name, output_dir):
    """
    Plots SHAP value contributions and feature group's values side by side.
    Saves each plot as a separate image file with the legend at the bottom.

    Args:
        shap_df (pd.DataFrame): DataFrame of SHAP values prepared for plotting.
        feature_values_df (pd.DataFrame): DataFrame of feature values prepared for plotting.
        group_name (str): Name of the group (e.g., 'global' or 'kg_class' value).
        output_dir (str): Directory to save the plots.
    """
    import matplotlib.pyplot as plt
    import os

    # Get feature groups based on specified rules
    feature_names = feature_values_df.columns.tolist()
    feature_groups = get_feature_groups(feature_names)

    # Create a mapping from group names to features
    group_to_features = {}
    for feature, group in feature_groups.items():
        group_to_features.setdefault(group, []).append(feature)

    for feature_group_name, features in group_to_features.items():
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8), sharex=True)

        # Plot SHAP contributions on the first subplot
        shap_df.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab20')
        axes[0].set_title('SHAP Value Contributions')
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Contribution')

        # Plot feature values for the specific group on the second subplot
        group_features = feature_values_df[features]
        group_features.plot(ax=axes[1])
        axes[1].set_title(f'Feature Values - Group: {feature_group_name}')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Feature Value')

        # Add a light dotted horizontal line at y=0 in the feature plot
        axes[1].axhline(0, linestyle='--', color='lightgray', linewidth=1)

        # Adjust legends
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

        # Adjust layout and save the figure
        title = f'ΔUHI Contribution and Feature Values by Hour - {group_name} - Feature Group: {feature_group_name}'
        plt.suptitle(title, y=1.02)
        plt.tight_layout()

        # Create a subdirectory for the current group if it doesn't exist
        group_dir = os.path.join(output_dir, group_name)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

        output_path = os.path.join(group_dir, f'shap_and_feature_values_{group_name}_{feature_group_name}.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Plot saved as '{output_path}' for group '{group_name}' and Feature Group '{feature_group_name}'.")

def plot_shap_and_feature_values(shap_df, feature_values_df, kg_classes, output_dir):
    """
    Plots SHAP value contributions and feature group's values side by side.
    Generates plots for global data and each KGMajorClass filtered data.
    Saves each plot as a separate image file with the legend at the bottom.

    Args:
        shap_df (pd.DataFrame): DataFrame of SHAP values prepared for plotting.
        feature_values_df (pd.DataFrame): DataFrame of feature values prepared for plotting.
        kg_classes (list): List of unique KGMajorClass values.
        output_dir (str): Directory to save the plots.
    """
    # Plot for global data (no filtering by kg_class)
    print("Generating global plots (no KGMajorClass filtering)...")
    shap_plot_df_global = prepare_plot_df(
        shap_df,
        feature_col='Feature',
        value_col='Value',
        group_cols=['local_hour']
    )

    feature_values_plot_df_global = feature_values_df.pivot_table(
        index='local_hour',
        columns='Feature',
        values='FeatureValue',
        aggfunc='mean'  # Adjust aggregation if necessary
    )

    plot_shap_and_feature_values_for_group(
        shap_plot_df_global,
        feature_values_plot_df_global,
        group_name='global',
        output_dir=output_dir
    )

    # Plot for each kg_class
    for kg_class in kg_classes:
        print(f"Generating plots for KGMajorClass '{kg_class}'...")
        shap_df_subset = shap_df[shap_df['KGMajorClass'] == kg_class]
        shap_plot_df = prepare_plot_df(
            shap_df_subset,
            feature_col='Feature',
            value_col='Value',
            group_cols=['local_hour']
        )

        feature_values_subset = feature_values_df[feature_values_df['KGMajorClass'] == kg_class]
        feature_values_plot_df = feature_values_subset.pivot_table(
            index='local_hour',
            columns='Feature',
            values='FeatureValue',
            aggfunc='mean'  # Adjust aggregation if necessary
        )

        plot_shap_and_feature_values_for_group(
            shap_plot_df,
            feature_values_plot_df,
            group_name=kg_class,
            output_dir=output_dir
        )

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    kg_major_classes = df_feature['KGMajorClass'].unique()
    plot_shap_and_feature_values(
        df_feature,
        feature_values_melted,
        kg_major_classes,
        output_dir
    )

if __name__ == "__main__":
    main()
