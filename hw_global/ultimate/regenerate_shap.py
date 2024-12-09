import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add lookup table reading (if needed, or use a more efficient method to get LaTeX labels)
lookup_df = pd.read_excel('/home/jguo/research/hw_global/Data/var_name_unit_lookup.xlsx')
lookup_dict = dict(zip(lookup_df['Variable'], lookup_df['LaTeX']))

def get_latex_label(feature_name):
    """
    Retrieves the LaTeX label for a given feature based on its feature group.
    """
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
    
    latex_label = lookup_dict.get(feature_group)
    
    if pd.isna(latex_label) or latex_label == '':
        latex_label = feature_group
    final_label = f"{symbol}{latex_label}".strip()
    return final_label

def get_long_name(var_name, df_daily_vars):
    """
    Retrieve the long name for a given variable name from the dataframe.
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
    Add a 'Long Name' column to the input dataframe.
    """
    input_df['Long Name'] = input_df[join_column].apply(lambda x: get_long_name(x, df_daily_vars))
    return input_df

def get_shap_feature_importance(shap_values, feature_names, df_daily_vars):
    """
    Calculate the SHAP feature importance.
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
    plt.gcf().set_size_inches(15, 10)
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path)
    plt.close()

def process_shap_values_for_experiment(experiment_name, exclude_features=None):
    """
    Process SHAP values for a given experiment. Generates summary and waterfall plots.
    """
    mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.error(f"Experiment '{experiment_name}' not found.")
        return

    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)
    if len(runs) == 0:
        logging.error(f"No runs found in experiment '{experiment_name}'.")
        return

    run_id = runs.iloc[0].run_id
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    artifact_uri = artifact_uri.replace("mlflow-artifacts:", "/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary/mlflow/mlartifacts")

    # Load data
    df_daily_vars_path = os.path.join(artifact_uri, "hourlyDataSchema.xlsx")
    shap_values_feather_path = os.path.join(artifact_uri, "shap_values_with_additional_columns.feather")

    if os.path.exists(df_daily_vars_path):
        df_daily_vars = pd.read_excel(df_daily_vars_path)
    else:
        logging.warning(f"df_daily_vars file not found at {df_daily_vars_path}. Continuing without it.")
        df_daily_vars = None

    if os.path.exists(shap_values_feather_path):
        shap_df = pd.read_feather(shap_values_feather_path)
        logging.info(f"Loaded shap_values from {shap_values_feather_path}")
        logging.info(f"shap_df shape: {shap_df.shape}")

        # Display column information
        logging.info("Column Information:")
        logging.info("========================================")
        for i, col in enumerate(shap_df.columns):
            logging.info(f"{i:5}  {col:36}  {str(shap_df.dtypes[col]):14} {shap_df[col].isnull().any()}")
    else:
        logging.error(f"shap_values_with_additional_columns.feather file not found at {shap_values_feather_path}")
        return

    # Determine time period
    if 'day' in experiment_name.lower():
        time_period = 'day'
    elif 'night' in experiment_name.lower():
        time_period = 'night'
    else:
        raise ValueError("Experiment name should contain either 'day' or 'night'.")

    # Prepare output directories
    post_process_shap_dir = os.path.join(artifact_uri, 'post_process_shap')
    feature_group_shap_dir = os.path.join(artifact_uri, 'feature_group_shap')
    os.makedirs(post_process_shap_dir, exist_ok=True)
    os.makedirs(feature_group_shap_dir, exist_ok=True)

    # Separate SHAP and non-SHAP columns
    shap_columns = [col for col in shap_df.columns if col.endswith('_shap')]
    non_shap_columns = [col for col in shap_df.columns if col not in shap_columns]

    shap_values_df = shap_df[shap_columns]
    feature_names = [col.replace('_shap', '') for col in shap_columns]
    shap_values = shap_values_df.values

    X = shap_df[[col.replace('_shap', '') for col in shap_columns if col.replace('_shap', '') in shap_df.columns]]

    # Calculate SHAP feature importance
    shap_feature_importance = get_shap_feature_importance(shap_values, feature_names, df_daily_vars)
    if exclude_features:
        shap_feature_importance = shap_feature_importance[~shap_feature_importance['Feature'].isin(exclude_features)]
    
    # Generate SHAP summary plot for individual features
    logging.info(f"Creating SHAP summary plot for individual features for {time_period}time...")
    if not X.empty:
        shap.summary_plot(
            shap_values[:, shap_feature_importance.index],
            X.iloc[:, shap_feature_importance.index],
            feature_names=shap_feature_importance['Feature'].tolist(),
            show=False
        )
    else:
        shap.summary_plot(
            shap_values,
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
            base_values=0,
            feature_names=shap_feature_importance['Feature'].tolist()
        ),
        show=False
    )
    plt.title(f'SHAP Value-based Importance Plot for {time_period.capitalize()}time')
    importance_output_path = f"post_process_{time_period}_shap_importance_plot.png"
    save_and_log_artifact(importance_output_path, post_process_shap_dir)
    plt.clf()

    # Calculate SHAP feature importance by group
    shap_feature_importance['Feature Group'] = shap_feature_importance['Feature'].apply(get_feature_group)
    shap_feature_importance_by_group = shap_feature_importance.groupby('Feature Group')['Importance'].sum().reset_index()
    total_importance = shap_feature_importance_by_group['Importance'].sum()
    shap_feature_importance_by_group['Percentage'] = (shap_feature_importance_by_group['Importance'] / total_importance) * 100
    shap_feature_importance_by_group.sort_values(by='Importance', ascending=False, inplace=True)

    # Normalize feature values
    X_normalized = X.copy()
    if not X.empty:
        for col in X.columns:
            max_val = X[col].max()
            min_val = X[col].min()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1e-8
            X_normalized[col] = (X[col] - min_val) / range_val

    # Calculate group-level SHAP values and feature values
    group_shap_values = []
    group_feature_values = []
    for group in shap_feature_importance_by_group['Feature Group']:
        group_features = shap_feature_importance['Feature'][shap_feature_importance['Feature Group'] == group].tolist()
        group_indices = [feature_names.index(feat) for feat in group_features]
        group_shap = shap_values[:, group_indices].sum(axis=1)
        group_shap_values.append(group_shap)
        if not X.empty:
            group_feature_value = X_normalized.iloc[:, group_indices].sum(axis=1).values
            group_feature_values.append(group_feature_value)
        else:
            group_feature_values.append(None)

    group_shap_values = np.array(group_shap_values).T
    if not X.empty:
        group_feature_values = np.array(group_feature_values).T
    else:
        group_feature_values = None

    # Save SHAP feature importance by group
    group_csv_path = f'{time_period}_shap_feature_importance_by_group.csv'
    shap_feature_importance_by_group.to_csv(os.path.join(feature_group_shap_dir, group_csv_path), index=False)

    # Create SHAP waterfall plot by feature group
    logging.info(f"Creating SHAP waterfall plot by feature group for {time_period}time...")
    plt.figure(figsize=(12, 0.5 * len(shap_feature_importance_by_group)))
    waterfall_values = shap_feature_importance_by_group['Percentage'].values
    base_value = 0
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

    # Create SHAP summary plot by feature group
    logging.info(f"Creating SHAP summary plot by feature group for {time_period}time...")
    plt.figure(figsize=(10, 8))
    if not X.empty:
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
    plt.clf()

    # Create percentage contribution plot
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
    plt.clf()

    # Save SHAP feature importance data to CSV
    shap_importance_path = os.path.join(post_process_shap_dir, f'{time_period}_shap_feature_importance.csv')
    shap_feature_importance.to_csv(shap_importance_path, index=False)
    logging.info(f"SHAP feature importance data saved to {shap_importance_path}")

def main():
    parser = argparse.ArgumentParser(description='Process SHAP values for a given experiment.')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the MLflow experiment')
    parser.add_argument('--exclude_features', nargs='*', default=[], help='List of features to exclude from SHAP summary plot')
    args = parser.parse_args()

    process_shap_values_for_experiment(args.experiment_name, args.exclude_features)

if __name__ == "__main__":
    main()