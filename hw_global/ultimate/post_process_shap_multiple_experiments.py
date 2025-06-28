import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow
import logging


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
        'hw_nohw_diff_': 'Δ',
        'Double_Differencing_': 'Δδ'
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


# Configure logging to display information, warnings, and errors with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def process_experiment(experiment_name, exclude_features=None):
    """
    Process a single experiment by loading the necessary data, calculating SHAP feature importance,
    and generating plots and CSV files for the results.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.error(f"Experiment '{experiment_name}' not found.")
        return None, None
    experiment_id = experiment.experiment_id

    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)
    if len(runs) == 0:
        logging.error(f"No runs found in experiment '{experiment_name}'. Please check the experiment name and make sure it contains runs.")
        return None, None

    run = runs.iloc[0]
    run_id = run.run_id

    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    artifact_uri = artifact_uri.replace("mlflow-artifacts:", "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts")

    df_daily_vars_path = os.path.join(artifact_uri, "hourlyDataSchema.xlsx")
    if os.path.exists(df_daily_vars_path):
        df_daily_vars = pd.read_excel(df_daily_vars_path)
        logging.info(f"Loaded df_daily_vars from {df_daily_vars_path}")
    else:
        logging.warning(f"df_daily_vars file not found at {df_daily_vars_path}. Continuing without it.")
        df_daily_vars = None

    if 'day' in experiment_name.lower():
        time_period = 'day'
    elif 'night' in experiment_name.lower():
        time_period = 'night'
    else:
        raise ValueError("Experiment name should contain either 'day' or 'night' to determine the time period.")

    shap_values_feather_path = os.path.join(artifact_uri, "shap_values_with_additional_columns.feather")
    if os.path.exists(shap_values_feather_path):
        shap_df = pd.read_feather(shap_values_feather_path)
        logging.info(f"Loaded shap_values from {shap_values_feather_path}")
    else:
        logging.error(f"shap_values_with_additional_columns.feather file not found at {shap_values_feather_path}")
        return None, None

    # Define non-SHAP columns to exclude
    non_shap_columns = ['global_event_ID', 'lon', 'lat', 'time', 'KGClass', 'KGMajorClass', 'UHI_diff', 'Estimation_Error']
    non_shap_columns = [col for col in non_shap_columns if col in shap_df.columns]

    # Extract KGMajorClass for partitioning before dropping non-SHAP columns
    if 'KGMajorClass' in shap_df.columns:
        kg_major_classes = shap_df['KGMajorClass'].unique()
    else:
        logging.error("KGMajorClass column not found in shap_values DataFrame.")
        kg_major_classes = []

    # Store KGMajorClass and other non-SHAP columns separately
    additional_columns_df = shap_df[non_shap_columns]

    # Extract SHAP values and feature names
    shap_values_df = shap_df.drop(columns=non_shap_columns)
    shap_values = shap_values_df.values
    feature_names = shap_values_df.columns.tolist()

    # Extract feature values from shap_df (assuming they are stored in the same columns)
    # If feature values are not available in shap_df, you may need to load them from another source
    # For this code, we'll assume that we need to load X_data.feather for feature values
    X_path = os.path.join(artifact_uri, "X_data.feather")
    if os.path.exists(X_path):
        X = pd.read_feather(X_path)
        logging.info(f"Loaded X data from {X_path}")
    else:
        logging.warning(f"X_data file not found at {X_path}. Cannot proceed with feature values.")
        X = None

    # Modify how we save and log artifacts to ensure they're in the correct directory
    post_process_shap_dir = os.path.join(artifact_uri, 'post_process_shap')
    feature_group_shap_dir = os.path.join(artifact_uri, 'feature_group_shap')
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
        X.iloc[:, shap_feature_importance.index] if X is not None else None,
        feature_names=shap_feature_importance['Feature'].tolist(),
        show=False
    )
    summary_output_path = f"post_process_{time_period}_shap_summary_plot.png"
    save_and_log_artifact(summary_output_path, post_process_shap_dir)
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
    plt.clf()

    # Calculate SHAP feature importance by group
    shap_feature_importance['Feature Group'] = shap_feature_importance['Feature'].apply(lambda x: get_feature_group(x))
    shap_feature_importance_by_group = shap_feature_importance.groupby('Feature Group')['Importance'].sum().reset_index()
    total_importance = shap_feature_importance_by_group['Importance'].sum()
    shap_feature_importance_by_group['Percentage'] = (shap_feature_importance_by_group['Importance'] / total_importance) * 100
    shap_feature_importance_by_group.sort_values(by='Importance', ascending=False, inplace=True)

    # Normalize feature values
    if X is not None:
        X_normalized = X.copy()
        for col in X.columns:
            max_val = X[col].max()
            min_val = X[col].min()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1e-8  # Avoid division by zero
            X_normalized[col] = (X[col] - min_val) / range_val
        logging.info("Feature values normalized using min-max scaling.")
    else:
        X_normalized = None

    # Calculate group-level SHAP values and normalized feature values
    group_shap_values = []
    group_feature_values = []
    for group in shap_feature_importance_by_group['Feature Group']:
        # Get features in this group
        group_features = shap_feature_importance['Feature'][shap_feature_importance['Feature Group'] == group].tolist()
        group_indices = [feature_names.index(feat) for feat in group_features]
        
        # Sum SHAP values for all features in the group
        group_shap = shap_values[:, group_indices].sum(axis=1)
        group_shap_values.append(group_shap)
        
        # Sum normalized feature values for all features in the group
        if X_normalized is not None:
            group_feature_value = X_normalized.iloc[:, group_indices].sum(axis=1).values
            group_feature_values.append(group_feature_value)
        else:
            group_feature_values.append(None)

    group_shap_values = np.array(group_shap_values).T  # Transform to shape (n_samples, n_groups)
    if X_normalized is not None:
        group_feature_values = np.array(group_feature_values).T  # Transform to shape (n_samples, n_groups)
    else:
        group_feature_values = None

    # Save SHAP feature importance by group
    group_csv_path = f'{time_period}_shap_feature_importance_by_group.csv'
    shap_feature_importance_by_group.to_csv(os.path.join(feature_group_shap_dir, group_csv_path), index=False)

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
    plt.clf()

    # Create and log SHAP summary plot by feature group
    logging.info(f"Creating SHAP summary plot by feature group for {time_period}time...")
    plt.figure(figsize=(10, 8))
    if X_normalized is not None:
        shap.summary_plot(
            group_shap_values,
            group_feature_values,
            feature_names=shap_feature_importance_by_group['Feature Group'].tolist(),
            show=False
        )
    else:
        shap.summary_plot(
            group_shap_values,
            feature_names=shap_feature_importance_by_group['Feature Group'].tolist(),
            show=False
        )
    plt.title(f'SHAP Summary Plot by Feature Group for {time_period.capitalize()}time')
    summary_group_output_path = f"post_process_{time_period}_shap_summary_plot_by_group.png"
    save_and_log_artifact(summary_group_output_path, feature_group_shap_dir)

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

    # Save SHAP feature importance data to CSV
    shap_importance_path = os.path.join(post_process_shap_dir, f'{time_period}_shap_feature_importance.csv')
    shap_feature_importance.to_csv(shap_importance_path, index=False)
    logging.info(f"SHAP feature importance data saved to {shap_importance_path}")

    # Now process by KGMajorClass
    kg_major_dir = os.path.join(artifact_uri, 'KGMajor')
    os.makedirs(kg_major_dir, exist_ok=True)

    for kg_major_class in kg_major_classes:
        class_indices = shap_df.index[shap_df['KGMajorClass'] == kg_major_class].tolist()
        if not class_indices:
            continue

        class_shap_values = shap_values[class_indices]
        if X is not None:
            class_X = X.iloc[class_indices]
        else:
            class_X = None

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
            class_X.iloc[:, shap_feature_importance_class.index] if class_X is not None else None,
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
        if class_X is not None:
            class_X_normalized = class_X.copy()
            for col in class_X.columns:
                max_val = class_X[col].max()
                min_val = class_X[col].min()
                range_val = max_val - min_val
                if range_val == 0:
                    range_val = 1e-8  # Avoid division by zero
                class_X_normalized[col] = (class_X[col] - min_val) / range_val
            logging.info(f"Feature values normalized for KGMajorClass '{kg_major_class}' using min-max scaling.")
        else:
            class_X_normalized = None

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
            if class_X_normalized is not None:
                group_feature_value_class = class_X_normalized.iloc[:, group_indices].sum(axis=1).values
                group_feature_values_class.append(group_feature_value_class)
            else:
                group_feature_values_class.append(None)

        group_shap_values_class = np.array(group_shap_values_class).T  # Transform to shape (n_samples, n_groups)
        if class_X_normalized is not None:
            group_feature_values_class = np.array(group_feature_values_class).T  # Transform to shape (n_samples, n_groups)
        else:
            group_feature_values_class = None

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
        if group_feature_values_class is not None:
            shap.summary_plot(
                group_shap_values_class,
                group_feature_values_class,
                feature_names=shap_feature_importance_by_group_class['Feature Group'].tolist(),
                show=False
            )
        else:
            shap.summary_plot(
                group_shap_values_class,
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

    return shap_feature_importance, shap_feature_importance_by_group

def main(prefix, hw_pct, exclude_features):
    """
    Main function to process multiple experiments based on the given prefix and hw_pct.
    Combines the SHAP feature importance data from all processed experiments and saves it to an Excel file.
    """
    mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")

    experiments = mlflow.search_experiments()
    filtered_experiments = [exp for exp in experiments if exp.name.startswith(prefix) and hw_pct in exp.name]

    all_shap_feature_importance = []
    all_shap_feature_importance_by_group = []

    for experiment in filtered_experiments:
        experiment_name = experiment.name
        logging.info(f"Processing experiment: {experiment_name}")

        shap_feature_importance, shap_feature_importance_by_group = process_experiment(experiment_name, exclude_features)
        if shap_feature_importance is not None:
            shap_feature_importance['ExperimentName'] = experiment_name
            all_shap_feature_importance.append(shap_feature_importance)

            shap_feature_importance_by_group['ExperimentName'] = experiment_name
            all_shap_feature_importance_by_group.append(shap_feature_importance_by_group)

    if len(all_shap_feature_importance) > 0:
        combined_shap_feature_importance = pd.concat(all_shap_feature_importance, ignore_index=True)
        
        # Sort by ExperimentName in ascending order and Percentage in descending order
        combined_shap_feature_importance.sort_values(by=['ExperimentName', 'Percentage'], ascending=[True, False], inplace=True)
        
        # Add a column for feature ranking within each experiment
        combined_shap_feature_importance['Rank'] = combined_shap_feature_importance.groupby('ExperimentName').cumcount() + 1
        
        output_path = f"combined_shap_feature_importance_{prefix}_{hw_pct}.xlsx"
        combined_shap_feature_importance.to_excel(output_path, index=False)
        logging.info(f"Combined shap_feature_importance saved to {output_path}")
    else:
        logging.warning("No shap_feature_importance data found in the processed experiments.")

    if len(all_shap_feature_importance_by_group) > 0:
        combined_shap_feature_importance_by_group = pd.concat(all_shap_feature_importance_by_group, ignore_index=True)
        
        # Save combined SHAP feature importance by group to Excel
        shap_feature_importance_by_group_excel_path = f"combined_shap_feature_importance_by_group_{prefix}_{hw_pct}.xlsx"
        combined_shap_feature_importance_by_group.to_excel(shap_feature_importance_by_group_excel_path, index=False)
        logging.info(f"Combined SHAP feature importance by group saved to {shap_feature_importance_by_group_excel_path}")
    else:
        logging.warning("No SHAP feature importance by group data found in the processed experiments.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SHAP feature importance for multiple experiments.')
    parser.add_argument('--prefix', type=str, default='KG_SHAP', help='Prefix for filtering experiments (default: KG_SHAP)')
    parser.add_argument('--hw_pct', type=str, default='HW98', help='hw_pct name for filtering experiments (default: HW98)')
    parser.add_argument('--exclude_features', nargs='*', default=[], help='List of features to exclude from SHAP summary plot')
    args = parser.parse_args()

    main(prefix=args.prefix, hw_pct=args.hw_pct, exclude_features=args.exclude_features)
