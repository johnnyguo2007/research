import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow
import logging

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

def process_experiment(experiment_name):
    """
    Process a single experiment by loading the necessary data, calculating SHAP feature importance,
    and generating plots and CSV files for the results.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.error(f"Experiment '{experiment_name}' not found.")
        return None
    experiment_id = experiment.experiment_id

    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)
    if len(runs) == 0:
        logging.error(f"No runs found in experiment '{experiment_name}'. Please check the experiment name and make sure it contains runs.")
        return None

    run = runs.iloc[0]
    run_id = run.run_id

    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    artifact_uri = artifact_uri.replace("mlflow-artifacts:", "/home/jguo/research/hw_global/mlartifacts")

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

    shap_values_path = os.path.join(artifact_uri, "shap_values.npy")
    if os.path.exists(shap_values_path):
        shap_values = np.load(shap_values_path)
        logging.info(f"Loaded shap_values from {shap_values_path}")
    else:
        logging.error(f"shap_values file not found at {shap_values_path}")
        return None

    feature_names_path = os.path.join(artifact_uri, "feature_names.txt")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f]
        logging.info(f"Loaded feature_names from {feature_names_path}")
    else:
        logging.error(f"feature_names file not found at {feature_names_path}")
        return None

    X_path = os.path.join(artifact_uri, "X_data.feather")
    if os.path.exists(X_path):
        X = pd.read_feather(X_path)
        logging.info(f"Loaded X data from {X_path}")
    else:
        logging.warning(f"X_data file not found at {X_path}. Continuing without it.")
        X = None

    shap_feature_importance = get_shap_feature_importance(shap_values, feature_names, df_daily_vars)

    # Create and save SHAP waterfall plot
    logging.info(f"Creating SHAP waterfall plot for {time_period}time...")
    plt.figure(figsize=(12, 0.5 * len(shap_feature_importance)))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_feature_importance['Importance'].values,
            base_values=shap_values[:, -1].mean(),
            data=X.iloc[0] if X is not None else None,
            feature_names=shap_feature_importance['Long Name'].tolist()
        ),
        show=False,
        max_display=len(shap_feature_importance)
    )
    plt.title(f'SHAP Waterfall Plot for {time_period.capitalize()}time')
    waterfall_output_path = os.path.join(artifact_uri, f"post_process_{time_period}_shap_waterfall_plot.png")
    plt.savefig(waterfall_output_path, dpi=300, bbox_inches='tight')
    logging.info(f"SHAP waterfall plot for {time_period}time saved to {waterfall_output_path}")
    plt.close()

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
    percentage_output_path = os.path.join(artifact_uri, f"post_process_{time_period}_percentage_contribution_plot.png")
    plt.savefig(percentage_output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Percentage contribution plot for {time_period}time saved to {percentage_output_path}")
    plt.close()

    # Save SHAP feature importance data to CSV
    shap_importance_path = os.path.join(artifact_uri, f'{time_period}_shap_feature_importance.csv')
    shap_feature_importance.to_csv(shap_importance_path, index=False)
    logging.info(f"SHAP feature importance data saved to {shap_importance_path}")

    return shap_feature_importance

def main(prefix, hw_pct):
    """
    Main function to process multiple experiments based on the given prefix and hw_pct.
    Combines the SHAP feature importance data from all processed experiments and saves it to an Excel file.
    """
    mlflow.set_tracking_uri(uri="http://192.168.4.85:8080")

    experiments = mlflow.search_experiments()
    filtered_experiments = [exp for exp in experiments if exp.name.startswith(prefix) and hw_pct in exp.name]

    all_shap_feature_importance = []

    for experiment in filtered_experiments:
        experiment_name = experiment.name
        logging.info(f"Processing experiment: {experiment_name}")

        shap_feature_importance = process_experiment(experiment_name)
        if shap_feature_importance is not None:
            shap_feature_importance['ExperimentName'] = experiment_name
            all_shap_feature_importance.append(shap_feature_importance)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SHAP feature importance for multiple experiments.')
    parser.add_argument('--prefix', type=str, default='KG_SHAP', help='Prefix for filtering experiments (default: KG_SHAP)')
    parser.add_argument('--hw_pct', type=str, default='HW98', help='hw_pct name for filtering experiments (default: HW98)')
    args = parser.parse_args()

    main(prefix=args.prefix, hw_pct=args.hw_pct)
