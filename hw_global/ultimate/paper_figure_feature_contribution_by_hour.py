import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import logging
import mlflow
import seaborn as sns
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

def replace_cold_with_continental(kg_main_group):
    if kg_main_group == 'Cold':
        return 'Continental'
    return kg_main_group

# Add lookup table reading
lookup_df = pd.read_excel('/home/jguo/research/hw_global/Data/var_name_unit_lookup.xlsx')
lookup_dict = dict(zip(lookup_df['Variable'], lookup_df['LaTeX']))

def get_latex_label(feature_name):
    """
    Retrieves the LaTeX label for a given feature based on its feature group.
    
    Args:
        feature_name (str): The name of the feature.
    
    Returns:
        str: The corresponding LaTeX label.
    """
    # Define mapping from prefixes to symbols
    prefix_to_symbol = {
        'delta_': '(Δ)',
        'hw_nohw_diff_': 'HW-NHW ',
        'Double_Differencing_': '(Δ)HW-NHW '
    }
    symbol = ''
    feature_group = feature_name
    for prefix in prefix_to_symbol.keys():
        if feature_name.startswith(prefix):
            feature_group = feature_name[len(prefix):]
            symbol = prefix_to_symbol[prefix]
            break
    # if feature_group == feature_name:
    #     feature_group += "_Level"
    
    # Get the LaTeX label from the lookup dictionary
    latex_label = lookup_dict.get(feature_group)
    
    # Use the original feature group if LaTeX label is not found
    if pd.isna(latex_label) or latex_label == '':
        latex_label = feature_group
    # Combine symbol and LaTeX label
    final_label = f"{symbol}{latex_label}".strip()
    return final_label

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
    
    # Extract base_value before dropping columns
    base_value = shap_df['base_value'].iloc[0] if 'base_value' in shap_df.columns else 0
    
    # Create a copy of the original data before dropping columns
    feature_values_df = shap_df.copy()
    
    # Drop specified columns
    columns_to_drop = ['global_event_ID', 'lon', 'lat', 'time', 'KGClass', 'KGMajorClass', 'base_value']
    shap_df = shap_df.drop(columns=columns_to_drop, errors='ignore')
    
    # Generate summary plots for global data
    summary_dir = os.path.join(output_dir, 'summary_plots')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Get SHAP columns and corresponding feature columns
    shap_cols = [col for col in shap_df.columns if col.endswith('_shap')]
    feature_cols = [col.replace('_shap', '') for col in shap_cols]
    
    # Generate summary plots for global data
    generate_summary_plots(
        shap_df[shap_cols],
        feature_values_df[feature_cols],
        summary_dir
    )
    
    # Generate summary plots for each KGMajorClass
    if 'KGMajorClass' in feature_values_df.columns:
        for kg_class in feature_values_df['KGMajorClass'].unique():
            kg_mask = feature_values_df['KGMajorClass'] == kg_class
            generate_summary_plots(
                shap_df[kg_mask][shap_cols],
                feature_values_df[kg_mask][feature_cols],
                summary_dir,
                kg_class
            )
    
    # Step 1: Group by 'local_hour' and 'KGMajorClass' and calculate mean of all shap value columns
    group_cols = ['local_hour', 'KGMajorClass']
    
    # Identify SHAP value columns: those ending with '_shap' and not in group_cols
    shap_cols = [col for col in shap_df.columns if col.endswith('_shap') and col not in [f"{gc}_shap" for gc in group_cols]]
    
    if not shap_cols:
        logging.warning("No SHAP columns found after excluding group-related columns.")
    
    # Group and calculate mean of the shap value columns
    df_grouped = shap_df.groupby(group_cols)[shap_cols].mean().reset_index()
    logging.info("Grouped by 'local_hour' and 'KGMajorClass' and calculated mean SHAP value columns.")
    
    # Step 2: Drop specified columns after grouping
    columns_to_drop = ['global_event_ID', 'lon', 'lat', 'time', 'KGClass', 'base_value']
    df_grouped = df_grouped.drop(columns=columns_to_drop, errors='ignore')
    logging.info(f"Dropped columns: {columns_to_drop}")
    
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
    
    return df_feature_group, df_feature, base_value

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


def save_plot_data(df, total_values, output_path, plot_type):
    """
    Save plot data to CSV files, including both individual values and totals.
    
    Args:
        df: DataFrame containing the plot data
        total_values: Series containing the total values
        output_path: Base path for the output file
        plot_type: String indicating the type of plot ('shap' or 'feature')
    """
    # Remove .png extension and add csv
    base_path = output_path.rsplit('.', 1)[0]
    
    # Create a copy of the DataFrame
    output_df = df.copy()
    
    # Add the total values as a new column
    output_df['Total'] = total_values
    
    # Save to CSV
    output_df.to_csv(f"{base_path}_{plot_type}_data.csv")
    
    logging.info(f"Saved {plot_type} data to {base_path}_{plot_type}_data.csv")

def plot_shap_stacked_bar(shap_df, title, output_path, color_mapping=None, return_fig=False, base_value=0):
    """
    Plots a standalone SHAP stacked bar plot (all features) with a mean SHAP value curve.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    if color_mapping:
        sorted_columns = sorted(shap_df.columns, key=lambda x: x)
        colors = [color_mapping.get(feature, '#333333') for feature in sorted_columns]
        shap_df = shap_df[sorted_columns]
        shap_df.plot(kind='bar', stacked=True, color=colors, ax=ax, bottom=base_value)  
    else:
        shap_df.plot(kind='bar', stacked=True, colormap='tab20', ax=ax, bottom=base_value)  
    
    # Calculate mean SHAP values and add base_value
    mean_shap = shap_df.sum(axis=1) + base_value
    
    # Plot the mean SHAP values as a line on the same axis
    mean_shap.plot(kind='line', color='black', marker='o', linewidth=2, ax=ax, label='Mean SHAP + Base Value')

    # Add base value line
    ax.axhline(y=base_value, color='red', linestyle='--', label=f'Base Value ({base_value:.3f})')

    # Get handles and labels, convert feature names to LaTeX labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label.startswith('Mean SHAP') or label.startswith('Base Value'):
            new_labels.append(label)
        else:
            new_labels.append(get_latex_label(label))
    
    ax.legend(handles, new_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

    plt.title(title)
    plt.xlabel('Hour of Day')
    ax.set_ylabel('Mean SHAP Value Contribution')
    plt.tight_layout()

    if return_fig:
        return fig, ax
    else:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Standalone SHAP stacked bar plot with mean curve saved at '{output_path}'.")

    # Save data before plotting
    save_plot_data(
        shap_df,  # Save original values without base_value adjustment
        mean_shap,
        output_path,
        'shap'
    )

def plot_shap_and_feature_values_for_group(shap_df, feature_values_df, group_name, output_dir, kg_class, color_mapping, base_value=0, show_total_feature_line=True):
    """
    Plots SHAP value contributions and feature group's values side by side.
    
    Args:
        shap_df: DataFrame containing SHAP values
        feature_values_df: DataFrame containing feature values
        group_name: Name of the feature group
        output_dir: Directory to save output
        kg_class: KGMajorClass name
        color_mapping: Dictionary mapping features to colors
        base_value: Base value for SHAP contributions (default: 0)
        show_total_feature_line: Whether to show total feature value line (default: True)
    """
    import matplotlib.pyplot as plt
    import os

    # Check if data is available
    if shap_df.empty or feature_values_df.empty:
        logging.warning(f"No data available for group '{group_name}' in KGMajorClass '{kg_class}'. Skipping.")
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
    # Calculate mean SHAP values for each feature at each hour
    mean_shap_df = shap_df.copy()
    mean_shap_df.plot(kind='bar', stacked=True, ax=axes[0], color=shap_colors)

    axes[0].set_title('Mean SHAP Value Contributions')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Mean Contribution')

    # Calculate and plot mean SHAP values on the same axis
    mean_shap = mean_shap_df.sum(axis=1)  # Sum across features for each hour
    mean_shap.plot(kind='line', color='black', marker='o', linewidth=2, 
                    ax=axes[0])

    # Get handles and labels for SHAP plot, convert feature names to LaTeX labels
    handles, labels = axes[0].get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label.startswith('Mean SHAP') or label.startswith('Base Value'):
            new_labels.append(label)
        else:
            new_labels.append(get_latex_label(label))
    axes[0].legend(handles, new_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

    # Plot feature values on axes[1]
    feature_values_df.plot(ax=axes[1], color=feature_colors)
    axes[1].set_title(f'Feature Values - Group: {get_latex_label(group_name)}')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Feature Value')
    axes[1].axhline(0, linestyle='--', color='lightgray', linewidth=1)

    # Add total feature values line if enabled and there are multiple features
    if show_total_feature_line and len(feature_values_df.columns) > 1:
        # Calculate and plot total feature values on the same axis
        total_features = feature_values_df.sum(axis=1)
        total_features.plot(kind='line', color='black', marker='o', linewidth=2, 
                          ax=axes[1], label='Total Feature Value')

    # Get handles and labels for feature values plot, convert feature names to LaTeX labels
    handles, labels = axes[1].get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label == 'Total Feature Value':
            new_labels.append(label)
        else:
            new_labels.append(get_latex_label(label))
    axes[1].legend(handles, new_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

    # Adjust layout and save the figure
    title = f'HW-NHW UHI Contribution and Feature Values by Hour - {get_latex_label(group_name)} - Climate Zone {replace_cold_with_continental(kg_class)}'
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved as '{output_path}' for group '{group_name}' and KGMajorClass '{kg_class}'.")

    # Calculate totals
    total_shap = shap_df.sum(axis=1)
    total_features = feature_values_df.sum(axis=1) if len(feature_values_df.columns) > 1 else pd.Series(0, index=feature_values_df.index)

    # Save data before plotting
    save_plot_data(
        shap_df, 
        total_shap,
        output_path,
        'shap'
    )
    save_plot_data(
        feature_values_df,
        total_features,
        output_path,
        'feature'
    )

def plot_shap_and_feature_values(df_feature, feature_values_melted, kg_classes, output_dir, base_value=0, show_total_feature_line=True):
    """
    Plots SHAP value contributions and feature group's values side by side.
    
    Args:
        df_feature: DataFrame containing feature data
        feature_values_melted: Melted DataFrame containing feature values
        kg_classes: List of KGMajorClasses
        output_dir: Directory to save output
        base_value: Base value for SHAP contributions (default: 0)
        show_total_feature_line: Whether to show total feature value line (default: True)
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
            output_dir=global_dir,
            kg_class='global',
            color_mapping=color_mapping,
            base_value=base_value,
            show_total_feature_line=show_total_feature_line
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
                output_dir=kg_class_dir,
                kg_class=kg_class,
                color_mapping=color_mapping,
                base_value=base_value,
                show_total_feature_line=show_total_feature_line
            )

# Define the function with 'Value' instead of 'Importance'
def plot_feature_group_stacked_bar(df, group_by_column, output_path, title, base_value=0):
    """
    Plots a stacked bar chart of mean feature group contributions with mean SHAP value line.
    """
    # Pivot and prepare data - calculate mean instead of sum
    pivot_df = df.pivot_table(
        index=group_by_column,
        columns='Feature Group',
        values='Value',
        aggfunc='mean',
        fill_value=0
    )
    
    # Calculate means including base_value
    mean_values = pivot_df.sum(axis=1) + base_value
    
    # Save data before plotting
    save_plot_data(
        pivot_df,
        mean_values,
        output_path,
        'group'
    )

    # Sort the index if necessary
    pivot_df = pivot_df.sort_index()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot stacked bars starting from base_value
    pivot_df.plot(kind='bar', stacked=True, ax=ax, bottom=base_value)

    # Plot mean values including base_value for feature group reports
    mean_values = pivot_df.sum(axis=1) + base_value
    mean_values.plot(color='black', marker='o', linewidth=2, 
                     ax=ax, label='Mean SHAP + Base Value')

    # Add base value line
    ax.axhline(y=base_value, color='red', linestyle='--', label=f'Base Value ({base_value:.3f})')

    # Get handles and labels, convert feature group names to LaTeX labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label.startswith('Mean SHAP') or label.startswith('Base Value'):
            new_labels.append(label)
        else:
            new_labels.append(get_latex_label(label))

    ax.legend(handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    ax.set_xlabel(group_by_column.replace('_', ' ').title())
    ax.set_ylabel('Mean SHAP Value Contribution')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def create_day_night_summary(df_feature_group, output_dir):
    """
    Creates a summary DataFrame of day and night contributions for each feature group and KGMajorClass.
    
    Args:
        df_feature_group: DataFrame containing feature group data
        output_dir: Directory to save the output summary
    """
    # Create empty lists to store results
    rows = []
    
    # Process global data first
    for period, hour_mask in [
        ("Day", lambda x: x.between(18, 19)),  # 18:00 to 19:59
        ("Night", lambda x: x.between(5, 6))  # 5:00 to 6:59
    ]:
        # Filter data for the current period
        period_data = df_feature_group[hour_mask(df_feature_group['local_hour'])]
        
        # Calculate mean for each feature group (changed from top 2 average to mean)
        group_means = period_data.groupby('Feature Group')['Value'].mean()
        
        # Add to rows with 'Global' as KGMajorClass
        rows.append({
            'KGMajorClass': 'Global',
            'Period': f"{period} Mean",  # Changed label from "Avg Top 2" to "Mean"
            **group_means.to_dict(),
            'Total': group_means.sum()
        })
    
    # Process each KGMajorClass
    for kg_class in df_feature_group['KGMajorClass'].unique():
        kg_data = df_feature_group[df_feature_group['KGMajorClass'] == kg_class]
        
        for period, hour_mask in [
            ("Day", lambda x: x.between(18, 19)),  # 18:00 to 19:59
            ("Night", lambda x: x.between(5, 6))  # 5:00 to 6:59
        ]:
            # Filter data for the current period
            period_data = kg_data[hour_mask(kg_data['local_hour'])]
            
            # Calculate mean for each feature group (changed from top 2 average to mean)
            group_means = period_data.groupby('Feature Group')['Value'].mean()
            
            # Add to rows
            rows.append({
                'KGMajorClass': kg_class,
                'Period': f"{period} Mean",  # Changed label from "Avg Top 2" to "Mean"
                **group_means.to_dict(),
                'Total': group_means.sum()
            })
    
    # Create DataFrame from rows
    summary_df = pd.DataFrame(rows)
    
    # Round all numeric columns to 6 decimal places
    numeric_cols = summary_df.select_dtypes(include=['float64', 'int64']).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(6)
    
    # Reorder columns to put Total at the end
    cols = ['KGMajorClass', 'Period'] + [col for col in summary_df.columns 
                                        if col not in ['KGMajorClass', 'Period', 'Total']] + ['Total']
    summary_df = summary_df[cols]
    
    # Custom sort order for KGMajorClass to put Global first
    kg_class_order = ['Global'] + sorted([x for x in summary_df['KGMajorClass'].unique() if x != 'Global'])
    summary_df['KGMajorClass'] = pd.Categorical(summary_df['KGMajorClass'], categories=kg_class_order, ordered=True)
    
    # Sort by KGMajorClass and Period
    summary_df = summary_df.sort_values(['KGMajorClass', 'Period'])
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'feature_group_day_night_summary.csv')
    summary_df.to_csv(output_path, index=False)
    
    # Create a pivot table for better visualization
    pivot_df = summary_df.pivot_table(
        index=['KGMajorClass', 'Period'],
        values=[col for col in summary_df.columns if col not in ['KGMajorClass', 'Period']],
        aggfunc='first'
    )
    
    # Reorder columns to put Total at the end in pivot_df
    cols = [col for col in pivot_df.columns if col != 'Total'] + ['Total']
    pivot_df = pivot_df[cols]
    
    # Save pivot table to CSV
    pivot_output_path = os.path.join(output_dir, 'feature_group_day_night_summary_pivot.csv')
    pivot_df.to_csv(pivot_output_path)
    
    print(f"Day/night summary saved to {output_path}")
    print(f"Day/night summary pivot table saved to {pivot_output_path}")
    
    return summary_df, pivot_df

def generate_summary_plots(shap_df, feature_values_df, output_dir, kg_class='Global'):
    """
    Generate feature summary plot, feature group summary plot and waterfall plot.
    
    Args:
        shap_df (pd.DataFrame): DataFrame containing SHAP values
        feature_values_df (pd.DataFrame): DataFrame containing feature values
        output_dir (str): Directory to save output plots
        kg_class (str): Name of the climate zone (default: 'Global')
    """
    logging.info(f"Starting generate_summary_plots with shap_df shape: {shap_df.shape} and feature_values_df shape: {feature_values_df.shape}")
    logging.info(f"SHAP columns: {shap_df.columns}")
    logging.info(f"Feature values columns: {feature_values_df.columns}")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature names (columns ending with '_shap')
    shap_cols = [col for col in shap_df.columns if col.endswith('_shap')]
    
    # Check if there are any SHAP values to plot
    if not shap_cols or shap_df.empty or feature_values_df.empty:
        logging.warning(f"No SHAP values or feature values available for {kg_class}. Skipping summary plots.")
        return
        
    feature_names = [col.replace('_shap', '') for col in shap_cols]
    
    # Convert SHAP values to numpy array
    shap_values = shap_df[shap_cols].values
    
    # Check if shap_values array is empty
    if shap_values.size == 0:
        print(f"Warning: Empty SHAP values array for {kg_class}. Skipping summary plots.")
        return
        
    # Convert feature values to numpy array
    feature_values = feature_values_df[feature_names].values
    
    try:
        # Generate feature summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            feature_values,
            feature_names=feature_names,
            show=False,
            plot_size=(12, 8)
        )
        plt.title(f'Feature Summary Plot - {kg_class}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_summary_plot_{kg_class}.png'), bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Generated feature summary plot for {kg_class}")
    except Exception as e:
        logging.error(f"Failed to generate feature summary plot for {kg_class}: {str(e)}")
        plt.close()
    
    # Get feature groups
    feature_groups = get_feature_groups(feature_names)
    
    # Calculate group-level SHAP values
    group_names = list(set(feature_groups.values()))
    group_shap_values = []
    group_feature_values = []
    
    for group in group_names:
        # Get features in this group
        group_features = [f for f, g in feature_groups.items() if g == group]
        
        # Get corresponding SHAP columns
        group_shap_cols = [f"{f}_shap" for f in group_features]
        
        # Sum SHAP values for features in the group
        group_shap = shap_df[group_shap_cols].values.sum(axis=1)
        group_shap_values.append(group_shap)
        
        # sum feature values for the group
        group_feat = feature_values_df[group_features].values.sum(axis=1)
        group_feature_values.append(group_feat)
    
    # Convert to numpy arrays
    group_shap_values = np.array(group_shap_values).T
    group_feature_values = np.array(group_feature_values).T
    
    # Check if group arrays are empty
    if group_shap_values.size == 0 or group_feature_values.size == 0:
        print(f"Warning: Empty group SHAP values or feature values array for {kg_class}. Skipping group summary plots.")
        return
    
    try:
        # Generate feature group summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            group_shap_values,
            group_feature_values,
            feature_names=group_names,
            show=False,
            plot_size=(12, 8)
        )
        plt.title(f'Feature Group Summary Plot - {kg_class}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_group_summary_plot_{kg_class}.png'), bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to generate feature group summary plot for {kg_class}: {str(e)}")
        plt.close()
    
    try:
        # Calculate mean absolute SHAP values for waterfall plot
        mean_abs_shap = np.abs(group_shap_values).mean(axis=0)
        total_shap = mean_abs_shap.sum()
        
        if total_shap == 0:
            print(f"Warning: Zero total SHAP value for {kg_class}. Skipping waterfall plot.")
            return
            
        shap_percentages = (mean_abs_shap / total_shap) * 100
        
        # Generate waterfall plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_percentages,
                base_values=0,
                data=None,
                feature_names=[get_latex_label(name) for name in group_names]
            ),
            show=False
        )
        plt.title(f'Feature Group Contribution Waterfall Plot - {kg_class}')
        plt.xlabel('Percentage Contribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_group_waterfall_plot_{kg_class}.png'), bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to generate waterfall plot for {kg_class}: {str(e)}")
        plt.close()

def get_experiment_and_run(experiment_name):
    """
    Retrieves the experiment and the latest run from MLflow.

    Args:
        experiment_name (str): Name of the MLflow experiment.

    Returns:
        tuple: The experiment ID and the latest run ID.
    """
    mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")
    logging.info("Set MLflow tracking URI")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.error(f"Experiment '{experiment_name}' not found.")
        return None, None

    experiment_id = experiment.experiment_id
    logging.info(f"Found experiment with ID: {experiment_id}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1
    )
    if len(runs) == 0:
        logging.error(f"No runs found in experiment '{experiment_name}'.")
        return experiment_id, None

    run = runs.iloc[0]
    run_id = run.run_id
    logging.info(f"Processing latest run with ID: {run_id}")

    return experiment_id, run_id

def prepare_shap_data(shap_df, feature_values_df, kg_class=None):
    """
    Prepares SHAP data for plotting by filtering and organizing values.
    
    Args:
        shap_df (pd.DataFrame): DataFrame containing SHAP values
        feature_values_df (pd.DataFrame): DataFrame containing feature values
        kg_class (str, optional): Climate zone to filter by
        
    Returns:
        tuple: (shap_plot_df, feature_values_plot_df)
    """
    # Filter by climate zone if specified
    if kg_class and kg_class != 'global':
        mask = feature_values_df['KGMajorClass'] == kg_class
        shap_df = shap_df[mask]
        feature_values_df = feature_values_df[mask]

    # Get SHAP columns and corresponding feature columns
    shap_cols = [col for col in shap_df.columns if col.endswith('_shap')]
    feature_cols = [col.replace('_shap', '') for col in shap_cols]

    return shap_df[shap_cols], feature_values_df[feature_cols]

def prepare_feature_group_data(df_feature, kg_class=None):
    """
    Prepares feature group data for plotting.
    
    Args:
        df_feature (pd.DataFrame): DataFrame containing feature data
        kg_class (str, optional): Climate zone to filter by
        
    Returns:
        pd.DataFrame: Processed feature group data
    """
    if kg_class and kg_class != 'global':
        df_feature = df_feature[df_feature['KGMajorClass'] == kg_class]
    
    return df_feature

def main():
    """Main function to process SHAP values and generate plots."""
    logging.info("Starting feature contribution analysis...")

    # Parse arguments
    args = parse_arguments()
    logging.info(f"Parsed command line arguments: {vars(args)}")

    # Get experiment and run
    experiment_id, run_id = get_experiment_and_run(args.experiment_name)
    if experiment_id is None or run_id is None:
        return

    # Setup paths
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    artifact_uri = artifact_uri.replace(
        "mlflow-artifacts:",
        "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts"
    )
    shap_values_feather_path = os.path.join(artifact_uri, "shap_values_with_additional_columns.feather")
    output_dir = os.path.join(artifact_uri, '24_hourly_plot')
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    shap_df = pd.read_feather(shap_values_feather_path)
    feature_values_df = shap_df.copy()
    base_value = shap_df['base_value'].iloc[0] if 'base_value' in shap_df.columns else 0

    # Clean data
    columns_to_drop = ['global_event_ID', 'lon', 'lat', 'time', 'KGClass', 'KGMajorClass', 'base_value']
    shap_df_cleaned = shap_df.drop(columns=columns_to_drop, errors='ignore')

    # Generate summary plots if requested
    if args.summary_plots_only:
        summary_dir = os.path.join(output_dir, 'summary_plots') 
        os.makedirs(summary_dir, exist_ok=True)
        
        # Get SHAP columns and corresponding feature columns
        shap_cols = [col for col in shap_df_cleaned.columns if col.endswith('_shap')]
        feature_cols = [col.replace('_shap', '') for col in shap_cols]
        
        # Generate summary plots for global data
        generate_summary_plots(
            shap_df_cleaned[shap_cols],
            feature_values_df[feature_cols],
            summary_dir
        )
        
        # Generate summary plots for each KGMajorClass
        if 'KGMajorClass' in feature_values_df.columns:
            for kg_class in feature_values_df['KGMajorClass'].unique():
                kg_mask = feature_values_df['KGMajorClass'] == kg_class
                generate_summary_plots(
                    shap_df_cleaned[kg_mask][shap_cols],
                    feature_values_df[kg_mask][feature_cols], 
                    summary_dir,
                    kg_class
                )
        return

    # Process SHAP values and generate feature group data
    df_feature_group, df_feature, base_value = report_shap_contribution_from_feather(
        shap_values_feather_path,
        output_dir,
        os.path.join(output_dir, 'shap_feature_group.feather'),
        os.path.join(output_dir, 'shap_pivot.feather'),
        os.path.join(output_dir, 'shap_feature.feather')
    )

    # Generate day/night summary if requested
    if args.day_night_summary_only:
        summary_df, pivot_df = create_day_night_summary(df_feature_group, output_dir)
        logging.info("\nFeature Group Day/Night Summary:")
        logging.info("\n" + str(pivot_df))
        return

    # Load and prepare feature values for plotting
    feature_values_melted = load_feature_values(shap_values_feather_path)
    
    # Generate plots for each climate zone
    kg_classes = ['global'] + df_feature['KGMajorClass'].unique().tolist()
    for kg_class in kg_classes:
        # Prepare data for current climate zone
        feature_group_data = prepare_feature_group_data(df_feature, kg_class)
        
        # Generate stacked bar plot
        plot_title = 'Global Feature Group Contribution by Hour' if kg_class == 'global' else \
                    f'Feature Group Contribution by Hour for {kg_class}'
        output_path = os.path.join(output_dir, f'feature_group_contribution_by_hour_{kg_class}.png')
        
        plot_feature_group_stacked_bar(
            feature_group_data,
            'local_hour',
            output_path,
            plot_title,
            base_value
        )
        
        # Generate SHAP and feature value plots
        plot_shap_and_feature_values_for_group(
            shap_df=feature_group_data.pivot_table(
                index='local_hour',
                columns='Feature',
                values='Value',
                fill_value=0
            ),
            feature_values_df=feature_values_melted[feature_values_melted['KGMajorClass'] == kg_class] if kg_class != 'global' else feature_values_melted,
            group_name=kg_class,
            output_dir=output_dir,
            kg_class=kg_class,
            color_mapping=dict(zip(feature_group_data['Feature'].unique(), sns.color_palette('tab20'))),
            base_value=base_value,
            show_total_feature_line=not args.hide_total_feature_line
        )

    # Create final summary
    summary_df, pivot_df = create_day_night_summary(df_feature_group, output_dir)
    logging.info("\nFeature Group Day/Night Summary:")
    logging.info("\n" + str(pivot_df))
    
    logging.info("Analysis completed successfully.")
    logging.info(f"All outputs have been saved to: {output_dir}")

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Report and plot SHAP value contributions by feature group and hour."
    )
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
    parser.add_argument(
        '--hide-total-feature-line',
        action='store_true',
        help='Hide the total feature value line in feature value plots.'
    )
    parser.add_argument(
        '--day-night-summary-only',
        action='store_true',
        default=False,
        help='Only generate the day/night summary without creating other plots.'
    )
    parser.add_argument(
        '--summary-plots-only',
        action='store_true',
        default=False,
        help='Only generate the summary plots then exit.'
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()
